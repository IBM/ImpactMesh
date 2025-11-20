
import logging
import albumentations as A
import numpy as np
from pathlib import Path
from torchgeo.datamodules import NonGeoDataModule
from .impactmesh_dataset import ImpactMeshDataset
from terratorch.datamodules.generic_pixel_wise_data_module import Normalize
from terratorch.datamodules.generic_multimodal_data_module import MultimodalNormalize, wrap_in_compose_is_list

# Dataset statisitcs
impactmesh_wildfire_means = {
    "S2L2A": [801.325, 861.655, 991.636, 1019.702, 1366.43, 2000.191, 2255.338, 2354.884, 2481.838, 2747.908, 2185.777, 1495.209],
    "S1RTC": [-9.838, -15.465],
    "DEM": [412.745],
}
impactmesh_wildfire_stds = {
    "S2L2A": [1960.514, 1732.936, 1494.812, 1384.473, 1385.129, 1309.367, 1322.601, 1352.448, 1336.39, 2379.374, 1145.593, 991.566],
    "S1RTC": [3.505, 3.422],
    "DEM": [354.58],
}

impactmesh_flood_means = {
    "S2L2A": [1223.128, 1251.355, 1423.443, 1408.984, 1786.818, 2448.316, 2685.642, 2745.795, 2817.936, 3194.081, 1964.659, 1399.317],
    "S1RTC": [-9.98, -15.968],
    "DEM": [141.786],
}
impactmesh_flood_stds = {
    "S2L2A": [2358.709, 2227.598, 2082.363, 2068.519, 2086.682, 2003.085, 2019.494, 2060.309, 2014.732, 2992.644, 1414.951, 1218.357],
    "S1RTC": [4.24, 4.105],
    "DEM": [189.363],
}


class ImpactMeshDataModule(NonGeoDataModule):
    """NonGeo LightningDataModule implementation for ImpactMesh."""

    def __init__(
        self,
        data_root: str,
        batch_size: int = 8,
        num_workers: int = 0,
        means: dict[str, list] = None,
        stds: dict[str, list] = None,
        train_split: str = None,
        val_split: str = None,
        test_split: str = None,
        predict_split: str = None,
        modalities: list[str] = None,
        label_dir: str = "MASK",
        image_grep: dict[str, str] = None,
        label_grep: str = None,
        timesteps: list[int] = None,
        concat_bands: bool = False,
        predict_data_root: str = None,
        train_transform: A.Compose | None | list[A.BasicTransform] = None,
        val_transform: A.Compose | None | list[A.BasicTransform] = None,
        test_transform: A.Compose | None | list[A.BasicTransform] = None,
        no_data_value: float = -9999.,
        no_data_replace: float = None,
        rgb_indices: dict[str, list[int]] = None,
        **kwargs,
    ) -> None:
        """
        Initialize ImpactMeshDataModule.

        Args:
            data_root (str): Root directory of dataset.
            batch_size (int): Batch size for DataLoaders. Default is 8.
            num_workers (int): Number of workers for data loading. Default is 0.
            means (dict[str, list], optional): Per-modality normalization means. Defaults to dataset statistics.
            stds (dict[str, list], optional): Per-modality normalization stds. Defaults to dataset statistics.
            train_split, val_split, test_split, predict_split (str, optional): Split identifiers.
                Otherwise, run all patches in data_root.
            modalities (list[str], optional): List of input modalities. Defaults to ["S2L2A", "S1RTC", "DEM"].
            label_dir (str): Directory name for labels. Default is "MASK".
            image_grep (dict[str, str], optional): Patterns for image file matching. Default to ImpactMesh pattern.
            label_grep (str, optional): Pattern for label file matching. Default to ImpactMesh pattern.
            timesteps (list[int], optional): Temporal indices to include. Defaults to [0, 1, 2, 3].
            concat_bands (bool): Whether to concatenate bands across modalities. If True, concatenate bands of all
                modalities, otherwise load samples as dict {<modality>: <torch.tensor>}. Defaults to False.
            predict_data_root (str, optional): Root for prediction data.
            train_transform, val_transform, test_transform: List of Albumentations transforms. Defaults to ToTensor.
            no_data_value (float): Value representing missing data. Default is -9999.
            no_data_replace (float, optional): Replacement for missing data (NaN and no_data_value). Default is 0.
            rgb_indices (dict[str, list[int]], optional): RGB band indices per modality.
            **kwargs: Additional arguments for parent class.
        """
        super().__init__(
            ImpactMeshDataset,
            batch_size=batch_size,
            num_workers=num_workers,
            **kwargs,
        )
        self.data_root = Path(data_root)
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.predict_split = predict_split
        self.modalities = modalities or ["S2L2A", "S1RTC", "DEM"]
        self.label_dir = label_dir
        self.predict_data_root = predict_data_root
        if rgb_indices is not None:
            self.rgb_indices = rgb_indices
        elif concat_bands:
            self.rgb_indices = rgb_indices or {"S2L2A": list(range(12)), "S1RTC": [-3, -2], "DEM": [-1]}
            self.rgb_indices = {m: v for m, v in self.rgb_indices.items() if m in self.modalities}
        else:
            self.rgb_indices = rgb_indices or {m: None for m in self.modalities}

        self.image_grep = image_grep or {"S2L2A": "_S2L2A.zarr.zip", "S1RTC": "_S1RTC.zarr.zip", "DEM": "_DEM.tif"}
        if label_grep is not None:
            self.label_grep = label_grep
        elif "flood" in str(self.data_root).lower():
            self.label_grep = "_annotation_flood.tif"
        elif "fire" in str(self.data_root).lower():
            self.label_grep = "_annotation_wildfire.tif"
        else:
            raise ValueError(f"Unknown label_grep: {label_grep}. "
                             f"Specify a label grep or include disaster type in data root {self.data_root}.")
        self.timesteps = timesteps or [0, 1, 2, 3]
        self.concat_bands = concat_bands
        self.no_data_value = no_data_value
        self.no_data_replace = no_data_replace if no_data_replace is not None else 0

        self.train_transform = wrap_in_compose_is_list(train_transform, image_modalities=self.modalities)
        self.val_transform = wrap_in_compose_is_list(val_transform, image_modalities=self.modalities)
        self.test_transform = wrap_in_compose_is_list(test_transform, image_modalities=self.modalities)

        if means is not None and stds is not None:
            pass
        elif "flood" in str(self.data_root).lower():
            means = {m: impactmesh_flood_means[m] for m in self.modalities}
            stds = {m: impactmesh_flood_stds[m] for m in self.modalities}
        elif "fire" in str(self.data_root).lower():
            means = {m: impactmesh_wildfire_means[m] for m in self.modalities}
            stds = {m: impactmesh_wildfire_stds[m] for m in self.modalities}
        else:
            raise ValueError(f"Specify means and std or include disaster type in data root {self.data_root}")

        if self.concat_bands:
            # Concatenate mean and std values
            self.means = np.concatenate([means[m] for m in self.modalities]).tolist()
            self.stds = np.concatenate([stds[m] for m in self.modalities]).tolist()

            self.aug = Normalize(self.means, self.stds)
        else:
            # Apply standardization per modality
            self.means = {m: means[m] for m in means.keys()}
            self.stds = {m: stds[m] for m in stds.keys()}

            self.aug = MultimodalNormalize(self.means, self.stds)


    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either fit, validate, test, or predict.
        """
        if stage in ["fit"]:
            self.train_dataset = ImpactMeshDataset(
                data_root=self.data_root,
                split_file=self.train_split,
                modalities=self.modalities,
                label_dir=self.label_dir,
                image_grep=self.image_grep,
                label_grep=self.label_grep,
                timesteps=self.timesteps,
                concat_bands=self.concat_bands,
                transform=self.train_transform,
                no_data_value=self.no_data_value,
                no_data_replace=self.no_data_replace,
                rgb_indices=self.rgb_indices,
                aug=self.aug,
            )
        if stage in ["fit", "validate"]:
            self.val_dataset = ImpactMeshDataset(
                data_root=self.data_root,
                split_file=self.val_split,
                modalities=self.modalities,
                label_dir=self.label_dir,
                image_grep=self.image_grep,
                label_grep=self.label_grep,
                timesteps=self.timesteps,
                concat_bands=self.concat_bands,
                transform=self.val_transform,
                no_data_value=self.no_data_value,
                no_data_replace=self.no_data_replace,
                rgb_indices=self.rgb_indices,
                aug=self.aug,
            )
        if stage in ["test"]:
            self.test_dataset = ImpactMeshDataset(
                data_root=self.data_root,
                split_file=self.test_split,
                modalities=self.modalities,
                label_dir=self.label_dir,
                image_grep=self.image_grep,
                label_grep=self.label_grep,
                timesteps=self.timesteps,
                concat_bands=self.concat_bands,
                transform=self.test_transform,
                no_data_value=self.no_data_value,
                no_data_replace=self.no_data_replace,
                rgb_indices=self.rgb_indices,
                aug=self.aug,
            )
        if stage in ["predict"]:
            if self.predict_data_root is None:
                logging.warning(f"predict_data_root is not specified, using default data_root {self.data_root}.")
            self.predict_dataset = ImpactMeshDataset(
                data_root=self.predict_data_root or self.data_root,
                split_file=self.predict_split,
                modalities=self.modalities,
                label_dir=None,
                image_grep=self.image_grep,
                label_grep=None,
                timesteps=self.timesteps,
                concat_bands=self.concat_bands,
                transform=self.test_transform,
                no_data_value=self.no_data_value,
                no_data_replace=self.no_data_replace,
                rgb_indices=self.rgb_indices,
                aug=self.aug,
            )
