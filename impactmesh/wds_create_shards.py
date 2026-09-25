"""Repackage the released ImpactMesh per-modality tars into WebDataset shards.

The dataset on HuggingFace stores one tar per modality per split. Training over it
directly means random reads across four archives per sample, which is slow on shared
or network storage. This script rewrites it into WebDataset shards that hold all
modalities of a patch in one contiguous record, so `WdsImpactMeshDataModule` can
stream them sequentially.

Source layout (as released on HuggingFace):
    <hf_root>/<split>/<MODALITY>.tar   with members  MODALITY/<patch_id><suffix>
    <hf_root>/split/impactmesh_<disaster>_test_holdout.txt

Output (combined-sample shards, one __key__=patch_id per sample):
    <out>/ImpactMesh-<Task>/{train,val,test}/<split>_shard_{000000..}.tar
    entries: s2l2a.npy s1rtc.npy dem.npy mask.npy meta.json

The test split is written as a single directory in which the first shards hold the
seen-event patches and the later ones the held-out events, so a subset is addressed
by a shard range rather than a separate copy of the data (see the example config).

Streams tar-to-tar: modality tars are opened together and iterated in the split's
patch_id order, holding one sample in memory at a time (no mass extraction).

Usage:
    hf download ibm-esa-geospatial/ImpactMesh-Flood --repo-type dataset \
        --local-dir data/hf/ImpactMesh-Flood
    python -m impactmesh.wds_create_shards \
        --hf-root data/hf/ImpactMesh-Flood --out data/shards --disaster flood
"""
import argparse
import io
import json
import math
import os
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import rasterio
import webdataset as wds
import xarray as xr
import zarr
from pyproj import Transformer

MODALITIES = ["S2L2A", "S1RTC", "DEM", "MASK"]
SUFFIX = {
    "S2L2A": "_S2L2A.zarr.zip",
    "S1RTC": "_S1RTC.zarr.zip",
    "DEM": "_DEM.tif",
}
MASK_SUFFIX = {"flood": "_annotation_flood.tif", "fire": "_annotation_wildfire.tif"}
DISASTER_DIR = {"flood": "ImpactMesh-Flood", "fire": "ImpactMesh-Fire"}
N_SHARDS = {"train": 128, "val": 8, "test_seen": 8, "test_holdout": 8}


def _decode_zarr_bands(raw):
    """zarr.zip bytes -> (bands ndarray, meta dict {timestamps, center_lat/lon})."""
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f:
        tmp = f.name
        f.write(raw)
    try:
        store = zarr.storage.ZipStore(tmp, mode="r")
        ds = xr.open_zarr(store)
        if "spatial_ref" in ds.data_vars:
            ds = ds.assign_coords({"spatial_ref": ds.spatial_ref})
        ds = ds.compute()
        bands = ds["bands"].values
        timestamps, lat, lon = _zarr_meta(ds)
        store.close()
        return bands, timestamps, lat, lon
    finally:
        os.remove(tmp)


def _zarr_meta(ds):
    """(timestamps, center_lat, center_lon) from a store's attrs (fallback: coords)."""
    a = ds.attrs
    ts = a.get("datetime")
    timestamps = ts.split(";") if isinstance(ts, str) else \
        [np.datetime_as_string(t, unit="s") if not np.isnat(t) else None
         for t in np.atleast_1d(ds["time"].values)]
    lat, lon = a.get("center_lat"), a.get("center_lon")
    if lat is None or lon is None:
        # reproject store center to lat/lon
        crs = ds.rio.crs
        xc, yc = float(ds["x"].values.mean()), float(ds["y"].values.mean())
        lon, lat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(xc, yc)
    return timestamps, float(lat), float(lon)


def _decode_tif(raw, bands=False):
    """tif bytes -> (ndarray, crs str or None, transform list or None)."""
    with rasterio.open(io.BytesIO(raw)) as src:
        arr = src.read() if bands else src.read(1)
        crs = str(src.crs) if src.crs else None
        transform = list(src.transform)[:6] if src.transform else None
        return arr, crs, transform


def _members_by_patch(tar_path):
    """Map patch_id -> TarInfo for one modality tar (member name = MOD/<id><suffix>)."""
    tf = tarfile.open(tar_path, "r")
    out = {}
    for m in tf.getmembers():
        if not m.isfile():
            continue
        name = os.path.basename(m.name)
        for suf in list(SUFFIX.values()) + list(MASK_SUFFIX.values()):
            if name.endswith(suf):
                out[name[: -len(suf)]] = m
                break
    return tf, out


def _build_sample(patch_id, tars, members, mask_suffix):
    """Read one patch's bytes from each modality tar and build the shard entry."""
    s2_raw = tars["S2L2A"].extractfile(members["S2L2A"][patch_id]).read()
    s1_raw = tars["S1RTC"].extractfile(members["S1RTC"][patch_id]).read()
    dem_raw = tars["DEM"].extractfile(members["DEM"][patch_id]).read()
    mask_raw = tars["MASK"].extractfile(members["MASK"][patch_id]).read()

    s2, s2_ts, lat, lon = _decode_zarr_bands(s2_raw)
    s1, s1_ts, _, _ = _decode_zarr_bands(s1_raw)  # same patch center as S2
    dem, _, _ = _decode_tif(dem_raw)
    mask, crs, transform = _decode_tif(mask_raw)  # georef from the mask tif

    # center is one point per patch (S1==S2); only timestamps differ per modality
    meta = {"patch_id": patch_id, "center_lat": lat, "center_lon": lon,
            "s2_timestamps": s2_ts, "s1_timestamps": s1_ts,
            "crs": crs, "transform": transform}

    sample = {"__key__": patch_id}
    for k, arr in [("s2l2a.npy", s2), ("s1rtc.npy", s1),
                   ("dem.npy", dem), ("mask.npy", mask)]:
        buf = io.BytesIO()
        np.save(buf, arr)
        sample[k] = buf.getvalue()
    sample["meta.json"] = json.dumps(meta).encode()
    return sample


def _write_split(patch_ids, tars, members, mask_suffix, out_dir, prefix,
                 n_shards, start_shard=0):
    out_dir.mkdir(parents=True, exist_ok=True)
    maxcount = math.ceil(len(patch_ids) / n_shards)
    pattern = (out_dir / f"{prefix}_shard_%06d.tar").as_posix()
    written = 0
    with wds.ShardWriter(pattern, maxcount=maxcount, start_shard=start_shard) as sink:
        for pid in patch_ids:
            if any(pid not in members[m] for m in MODALITIES):
                missing = [m for m in MODALITIES if pid not in members[m]]
                print(f"skip {pid}: missing {missing}")
                continue
            sink.write(_build_sample(pid, tars, members, mask_suffix))
            written += 1
    print(f"{prefix}: wrote {written} samples in <= {maxcount}/shard "
          f"(shards from {start_shard})")


def create_shards(hf_root, out, disaster):
    hf_root, out = Path(hf_root), Path(out)
    mask_suffix = MASK_SUFFIX[disaster]
    SUFFIX["MASK"] = mask_suffix
    holdout_file = hf_root / "split" / f"impactmesh_{disaster}_test_holdout.txt"
    holdout = set(holdout_file.read_text().split()) if holdout_file.exists() else set()
    ds_dir = DISASTER_DIR[disaster]

    for src_split in ["train", "val", "test"]:
        if not all((hf_root / src_split / f"{m}.tar").exists() for m in MODALITIES):
            # allows converting a partial download, e.g. --include "val/*" only
            print(f"Skipping {src_split}: no modality tars in {hf_root / src_split}")
            continue
        tars, members = {}, {}
        for m in MODALITIES:
            tars[m], members[m] = _members_by_patch(hf_root / src_split / f"{m}.tar")
        # patch_id order = sorted union of keys present in the driving (S2L2A) tar
        all_ids = sorted(members["S2L2A"].keys())

        if src_split == "test":
            # All 16 test shards in one test/ dir: 0-7 seen, 8-15 holdout, so a
            # full-test run reads test-{0..15} and seen/holdout are shard ranges.
            seen = [p for p in all_ids if p not in holdout]
            hold = [p for p in all_ids if p in holdout]
            out_dir = out / ds_dir / "test"
            _write_split(seen, tars, members, mask_suffix, out_dir, "test",
                         N_SHARDS["test_seen"], start_shard=0)
            _write_split(hold, tars, members, mask_suffix, out_dir, "test",
                         N_SHARDS["test_holdout"], start_shard=N_SHARDS["test_seen"])
        else:
            _write_split(all_ids, tars, members, mask_suffix,
                         out / ds_dir / src_split, src_split, N_SHARDS[src_split])
        for tf in tars.values():
            tf.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hf-root", required=True,
                    help="Directory holding the downloaded <split>/<MODALITY>.tar files.")
    ap.add_argument("--out", required=True, help="Directory to write the shards into.")
    ap.add_argument("--disaster", required=True, choices=["flood", "fire"])
    args = ap.parse_args()
    create_shards(args.hf_root, args.out, args.disaster)
