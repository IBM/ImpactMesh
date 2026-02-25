
import logging
import time

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import torch
import zarr
import rasterio
import warnings
from pathlib import Path
from typing import Callable
from torchgeo.datasets import NonGeoDataset
from terratorch.datasets.transforms import MultimodalTransforms, MultimodalToTensor
from .plotting_utils import plot_modality, red

class ImpactMeshDataset(NonGeoDataset):
    """ "
    Pytorch Dataset class to load samples from the ImpactMesh dataset.
    """

    def __init__(
        self,
        data_root: str | Path,
        split_file: str = None,
        modalities: list[str] = None,
        label_dir: str | None = "MASK",
        image_grep: dict[str, str] = None,
        label_grep: str | None = None,
        timesteps: list[int] = None,
        concat_bands: bool = False,
        transform=None,
        no_data_value: float = -9999.,
        no_data_replace: float = None,
        rgb_indices: dict[str, list[int]] = None,
        aug: Callable = None,
    ):
        """
        Build ImpactMesh dataset. See ImpactMeshDataModule for parameter descriptions.
        """
        super().__init__()
        data_root = Path(data_root)
        self.modalities = modalities or ["S2L2A", "S1RTC", "DEM"]
        self.data_root = {m: data_root / m for m in self.modalities}
        self.label_dir = data_root / label_dir if label_dir is not None else None
        self.timesteps = timesteps or [0, 1, 2, 3]
        self.image_grep = image_grep or {"S2L2A": "_S2L2A.zarr.zip", "S1RTC": "_S1RTC.zarr.zip", "DEM": "_DEM.tif"}
        self.label_grep = label_grep
        self.concat_bands = concat_bands
        self.no_data_value = no_data_value
        self.no_data_replace = no_data_replace if no_data_replace is not None else np.nan
        self.rgb_indices = rgb_indices
        self.aug = aug

        if split_file is not None:
            with open(split_file, "r") as f:
                # Load prefix from split file
                samples = f.readlines()
                self.samples = [s.strip() for s in samples]
        else:
            folder = self.data_root[self.modalities[0]]
            grep = self.image_grep[self.modalities[0]].strip("*")
            logging.warning(f"Split file missing! Loading all samples ids in {folder}")
            # Load samples prefix from label files
            files = list(folder.glob("*" + grep))
            self.samples = [s.name.split(grep)[0] for s in files]
            if len(self.samples) == 0:
                raise ValueError(f"No samples found in {folder} with grep {grep}")

        # If no transform is given, apply only to transform to torch tensor
        if isinstance(transform, A.Compose):
            self.transform = MultimodalTransforms(transform, non_image_modalities=[])
        elif transform is None:
            # Modality-specific default transforms
            self.transform = MultimodalToTensor(self.modalities)
        else:
            raise TypeError(f"transform must be an instance of Albumentations.Compose")


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        patch_id = self.samples[item]
        # Load data
        output = {}
        for modality in self.modalities:
            file_path = self.data_root[modality] / f"{patch_id}{self.image_grep[modality]}"
            if not file_path.exists():
                raise ValueError(f"{file_path} does not exist")
            elif str(file_path).endswith(".zarr.zip"):
                data = zarr.open_consolidated(file_path, mode="r")
                sample = data["bands"][...]
            elif str(file_path).endswith(".tif"):
                with rasterio.open(file_path, "r") as f:
                    sample = f.read(1)
            else:
                raise ValueError(f"Unknown type {file_path}.")

            # Add dims
            if sample.ndim == 2:
                sample = sample.reshape(1, 1, *sample.shape)
            elif sample.ndim == 3:
                sample = sample.reshape(1, *sample.shape)
            elif not sample.ndim == 4:
                raise ValueError(f"Unknown sample shape {sample.shape}, expected 2D, 3D or 4D (file {file_path}).")

            # Select timestamps
            if sample.shape[0] < len(self.timesteps):
                # Repeat inputs for all timestamps
                if sample.shape[0] != 1:
                    raise ValueError(f"Unexpected shape {sample.shape} in file {file_path}.")
                sample = sample.repeat(len(self.timesteps), axis=0)
            elif sample.shape[0] > len(self.timesteps):
                # Subsample time steps
                sample = sample[self.timesteps]

            # Reshape sample to channel last [time, H, W, channel] as expected by Albumentations
            sample = sample.transpose(0, 2, 3, 1)

            if len(self.timesteps) == 1:
                # Select single image
                sample = sample[0]

            # Convert to float
            sample = sample.astype(np.float32)

            # Replace no data
            sample[sample == self.no_data_value] = np.nan
            sample = np.nan_to_num(sample, nan=self.no_data_replace)

            output[modality] = sample

        if self.label_dir is not None:
            label_path = self.label_dir / f"{patch_id}{self.label_grep}"
            if not label_path.exists():
                raise ValueError(f"{label_path} does not exist")
            else:
                with rasterio.open(label_path, "r") as f:
                    output['mask'] = f.read(1)

        if self.transform:
            output = self.transform(output)

        if self.concat_bands:
            # Concatenate bands of all modalities
            output["image"] = torch.cat([output.pop(m) for m in self.modalities], dim=0)
        else:
            # Tasks expect data to be stored in "image", moving modalities to image dict
            output["image"] = {m: output.pop(m) for m in self.modalities}

        if "mask" in output:
            output['mask'] = output['mask'].long()

        if "DEM" in self.modalities:
            # TerraTorch expects a tif file as input to copy the metadata. Only works with DEM
            output['filename'] = str(self.data_root["DEM"] / f"{patch_id}{self.image_grep['DEM']}")

        return output

    def plot(self, sample, suptitle=None, **kwargs):
        target = sample.pop("mask", None)
        prediction = sample.pop("prediction", None)
        suptitle = suptitle or sample.pop("filename", None)
        # Add batch dim for denormalizing
        sample = {k: v.unsqueeze(0) for k, v in sample.items()}
        sample = self.aug(sample, denormalize=True)
        images = {}
        # Code currently expects all bands fix fixed plotting. Needs to be updated to handle subsets of bands
        if "image" in sample:
            # Concat
            for mod, indices in self.rgb_indices.items():
                if max(indices) > sample["image"].shape[1]:
                    warnings.warn(f"RGB indices {indices}, but sample has only {sample['image'].shape[0]} channels.")
                    continue
                images[mod] = sample["image"][0, indices]
        else:
            for mod, indices in self.rgb_indices.items():
                images[mod] = sample[mod][0, indices] if indices is not None else sample[mod][0]

        rows = len(images)
        num_images = len(self.timesteps)
        cols = num_images + int(target is not None) + int(prediction is not None)

        fig, ax = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        if rows == 1:
            ax = [ax]

        for i, (mod, img) in enumerate(images.items()):
            # timeseries = timeseries.reshape(num_images, -1, *timeseries.shape[-2:])
            if num_images == 1:
                plot_modality(mod, img, ax=ax[i][0])
            else:
                for t in range(num_images):
                    plot_modality(mod, img[:, t], ax=ax[i][t])
            ax[i][0].set_title(mod)

        i = rows // 2
        if target is not None:
            # Plot base image
            if num_images == 1:
                plot_modality(self.modalities[0], images[self.modalities[0]], ax=ax[i][num_images])
            else:
                plot_modality(self.modalities[0], images[self.modalities[0]][:, -1], ax=ax[i][num_images])

            ax[i][num_images].imshow(target.detach().cpu().numpy(),
                                     cmap=red, vmin=-1, vmax=1, interpolation="nearest", alpha=0.7)
            ax[i][num_images].set_title("Target")

        if prediction is not None:
            ax_id = num_images + int(target is not None)
            # Plot base image
            if num_images == 1:
                plot_modality(self.modalities[0], images[self.modalities[0]], ax=ax[i][ax_id])
            else:
                plot_modality(self.modalities[0], images[self.modalities[0]][:, -1], ax=ax[i][ax_id])

            ax[i][ax_id].imshow(prediction.detach().cpu().numpy(),
                                cmap=red, vmin=-1, vmax=1, interpolation="nearest", alpha=0.7)
            ax[i][ax_id].set_title("Prediction")

        for i in range(rows):
            for j in range(cols):
                ax[i][j].axis('off')

        if suptitle:
            plt.suptitle(suptitle)

        plt.tight_layout()
        return fig
