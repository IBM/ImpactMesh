"""WebDataset decode + sample processing for ImpactMesh WDS shards.

Yields the SAME sample dict as ImpactMeshDataset.__getitem__ so models/configs/
normalization are unchanged. The processing logic below is a self-contained copy
of the format-agnostic tail of ImpactMeshDataset.__getitem__ (kept separate so the
existing path-based dataset is not touched).
"""
import io
import json

import numpy as np
import torch

# shard extension -> canonical modality name
_EXT2MOD = {"s2l2a.npy": "S2L2A", "s1rtc.npy": "S1RTC", "dem.npy": "DEM"}


def decode_sample(sample):
    """Raw WDS sample (bytes) -> {modality: ndarray, 'mask': ndarray, 'meta': dict, '__key__': str}."""
    out = {"__key__": sample["__key__"], "meta": json.loads(sample["meta.json"])}
    for ext, mod in _EXT2MOD.items():
        if ext in sample:
            out[mod] = np.load(io.BytesIO(sample[ext]))
    out["mask"] = np.load(io.BytesIO(sample["mask.npy"]))
    return out


def process_sample(decoded, modalities, timesteps, no_data_value, no_data_replace,
                    transform, concat_bands):
    """Copy of ImpactMeshDataset.__getitem__ lines ~102-162 (numpy arrays in)."""
    output = {}
    for modality in modalities:
        sample = decoded[modality]
        # Add dims -> 4D (time, band, H, W)
        if sample.ndim == 2:
            sample = sample.reshape(1, 1, *sample.shape)
        elif sample.ndim == 3:
            sample = sample.reshape(1, *sample.shape)
        elif sample.ndim != 4:
            raise ValueError(f"Unknown sample shape {sample.shape} for {modality}.")

        # Select timesteps
        if sample.shape[0] < len(timesteps):
            if sample.shape[0] != 1:
                raise ValueError(f"Unexpected shape {sample.shape} for {modality}.")
            sample = sample.repeat(len(timesteps), axis=0)
        elif sample.shape[0] > len(timesteps):
            sample = sample[timesteps]

        # Channel last (time, H, W, band) for Albumentations
        sample = sample.transpose(0, 2, 3, 1)
        if len(timesteps) == 1:
            sample = sample[0]

        sample = sample.astype(np.float32)
        sample[sample == no_data_value] = np.nan
        sample = np.nan_to_num(sample, nan=no_data_replace)
        output[modality] = sample

    output["mask"] = decoded["mask"]

    if transform:
        output = transform(output)

    if concat_bands:
        output["image"] = torch.cat([output.pop(m) for m in modalities], dim=0)
    else:
        output["image"] = {m: output.pop(m) for m in modalities}

    output["mask"] = output["mask"].long()
    # WDS has no on-disk tif; filename carries the patch id. crs/transform stay in
    # the shard meta.json for a deferred georeferenced export (not in the batch, to
    # avoid default_collate choking on nested lists).
    output["filename"] = decoded["__key__"]
    return output


def make_transform(modalities, timesteps, no_data_value, no_data_replace,
                   transform, concat_bands):
    """Return a wds.map fn: decoded sample -> processed sample dict."""
    def _fn(decoded):
        return process_sample(decoded, modalities, timesteps, no_data_value,
                              no_data_replace, transform, concat_bands)
    return _fn
