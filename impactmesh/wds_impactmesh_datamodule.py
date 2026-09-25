"""WebDataset LightningDataModule for ImpactMesh (streaming shards).

Yields the same batch dict as ImpactMeshDataModule, so tasks/configs/normalization
are unchanged. Not a NonGeoDataModule subclass: WDS is an IterableDataset with no
__len__/random access, so the map-style dataloader factory does not apply.
"""
import albumentations as A
import numpy as np
import webdataset as wds
from lightning.pytorch import LightningDataModule
from torch.utils.data._utils.collate import default_collate
from terratorch.datasets.transforms import MultimodalTransforms, MultimodalToTensor
from terratorch.datamodules.generic_pixel_wise_data_module import Normalize
from terratorch.datamodules.generic_multimodal_data_module import (
    MultimodalNormalize, wrap_in_compose_is_list)

from .impactmesh_datamodule import datasets_stats
from .wds_impactmesh_dataset import decode_sample, make_transform


def _wrap_transform(transform, modalities):
    """Match ImpactMeshDataset: A.Compose -> MultimodalTransforms, else ToTensor."""
    if isinstance(transform, A.Compose):
        return MultimodalTransforms(transform, non_image_modalities=[])
    return MultimodalToTensor(modalities)


class WdsImpactMeshDataModule(LightningDataModule):
    def __init__(
        self,
        train_urls: str,
        val_urls: str,
        test_urls: dict[str, str],
        train_epoch_size: int,
        batch_size: int = 8,
        num_workers: int = 4,
        means: dict[str, list] = None,
        stds: dict[str, list] = None,
        modalities: list[str] = None,
        timesteps: list[int] = None,
        concat_bands: bool = False,
        train_transform: A.Compose | None | list[A.BasicTransform] = None,
        val_transform: A.Compose | None | list[A.BasicTransform] = None,
        test_transform: A.Compose | None | list[A.BasicTransform] = None,
        no_data_value: float = -9999.,
        no_data_replace: float | None = 0.,
        disaster: str = None,
        shuffle_buffer: int = 1000,
    ) -> None:
        super().__init__()
        self.train_urls = train_urls
        self.val_urls = val_urls
        self.test_urls = test_urls
        self.train_epoch_size = train_epoch_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.modalities = modalities or ["S2L2A", "S1RTC", "DEM"]
        self.timesteps = timesteps or [0, 1, 2, 3]
        self.concat_bands = concat_bands
        self.no_data_value = no_data_value
        self.no_data_replace = no_data_replace if no_data_replace is not None else 0.
        self.shuffle_buffer = shuffle_buffer

        disaster = disaster or ("flood" if "flood" in str(train_urls).lower() else "fire")
        self.train_transform = _wrap_transform(
            wrap_in_compose_is_list(train_transform, image_modalities=self.modalities), self.modalities)
        self.val_transform = _wrap_transform(
            wrap_in_compose_is_list(val_transform, image_modalities=self.modalities), self.modalities)
        self.test_transform = _wrap_transform(
            wrap_in_compose_is_list(test_transform, image_modalities=self.modalities), self.modalities)

        if means is None or stds is None:
            if disaster not in datasets_stats:
                raise ValueError(
                    f"Unknown disaster type {disaster!r}, expected one of "
                    f"{list(datasets_stats)} (or pass means and stds explicitly)."
                )
            means = datasets_stats[disaster]["means"]
            stds = datasets_stats[disaster]["stds"]
        means = {m: means[m] for m in self.modalities}
        stds = {m: stds[m] for m in self.modalities}

        if self.concat_bands:
            self.aug = Normalize(np.concatenate([means[m] for m in self.modalities]).tolist(),
                                 np.concatenate([stds[m] for m in self.modalities]).tolist())
        else:
            self.aug = MultimodalNormalize(means, stds)

    def _pipeline(self, urls, transform, shuffle):
        steps = [
            wds.ResampledShards(urls) if shuffle else wds.SimpleShardList(urls),
            wds.split_by_node,
            wds.split_by_worker,
            wds.tarfile_to_samples(),
            wds.map(decode_sample),
        ]
        if shuffle:
            steps.append(wds.shuffle(self.shuffle_buffer))
        steps.append(wds.map(make_transform(
            self.modalities, self.timesteps, self.no_data_value,
            self.no_data_replace, transform, self.concat_bands)))
        steps.append(wds.batched(self.batch_size, collation_fn=default_collate,
                                 partial=not shuffle))
        return wds.DataPipeline(*steps)

    def _loader(self, pipeline):
        return wds.WebLoader(pipeline, batch_size=None, num_workers=self.num_workers)

    def on_after_batch_transfer(self, batch, dataloader_idx):
        batch.pop("__key__", None)  # wds.batched re-adds it; keep it out of the model forward
        if self.trainer:
            return self.aug(batch)
        return batch

    def train_dataloader(self):
        pipe = self._pipeline(self.train_urls, self.train_transform, shuffle=True)
        # ResampledShards is infinite and NOT split across ranks, so divide by world_size
        # too — else every GPU pulls a full epoch and steps/epoch are world_size too many.
        world = self.trainer.world_size if self.trainer else 1
        return self._loader(pipe).with_epoch(self.train_epoch_size // (self.batch_size * world))

    def val_dataloader(self):
        return self._loader(self._pipeline(self.val_urls, self.val_transform, shuffle=False))

    def test_dataloader(self):
        # Lightning zips these loaders against the task's test_dataloaders_names
        # positionally. --ckpt_path restores that list from the checkpoint's saved
        # hyper_parameters (a stale order there wins over both config and CLI), so
        # make test_urls the single source of truth. configure_metrics() ran in the
        # task's __init__ and already baked the names into the test/<name>/ metric
        # prefixes, so it has to be re-run after correcting hparams.
        if self.trainer:
            task = self.trainer.lightning_module
            if task.hparams.get("test_dataloaders_names") != list(self.test_urls):
                task.hparams["test_dataloaders_names"] = list(self.test_urls)
                task.configure_metrics()
                task.test_metrics.to(task.device)
        loaders = [self._loader(self._pipeline(u, self.test_transform, shuffle=False))
                   for u in self.test_urls.values()]
        return loaders if len(loaders) > 1 else loaders[0]
