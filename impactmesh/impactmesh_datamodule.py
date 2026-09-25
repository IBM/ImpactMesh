import logging
import albumentations as A
import numpy as np
from pathlib import Path
from torchgeo.datamodules import NonGeoDataModule
from .impactmesh_dataset import ImpactMeshDataset
from terratorch.datamodules.generic_pixel_wise_data_module import Normalize
from terratorch.datamodules.generic_multimodal_data_module import (
    MultimodalNormalize,
    wrap_in_compose_is_list,
)

datasets_stats = {
    "fire": {
        "means": {
            "S2L2A": [
                656.307,
                726.342,
                922.971,
                1000.263,
                1387.474,
                2109.49,
                2388.396,
                2505.595,
                2627.147,
                2817.856,
                2319.783,
                1578.35,
            ],
            "S1RTC": [-10.132, -15.81],
            "DEM": [456.555],
        },
        "stds": {
            "S2L2A": [
                1396.137,
                1280.788,
                1170.872,
                1167.83,
                1176.989,
                1146.266,
                1186.601,
                1216.544,
                1200.161,
                1910.06,
                1048.237,
                905.556,
            ],
            "S1RTC": [3.677, 3.682],
            "DEM": [387.825],
        },
        "label_grep": "_annotation_wildfire.tif",
        "num_classes": 2,
        "class_weights": [0.217, 0.783],
    },
    "flood": {
        "means": {
            "S2L2A": [
                1460.264,
                1499.629,
                1663.836,
                1651.188,
                2056.615,
                2747.754,
                2990.725,
                3074.264,
                3130.254,
                3578.565,
                2111.968,
                1501.892,
            ],
            "S1RTC": [-9.871, -15.993],
            "DEM": [154.914],
        },
        "stds": {
            "S2L2A": [
                2587.746,
                2477.497,
                2315.289,
                2303.498,
                2283.186,
                2087.282,
                2048.065,
                2099.017,
                2008.163,
                3010.72,
                1332.457,
                1162.548,
            ],
            "S1RTC": [4.028, 3.899],
            "DEM": [368.796],
        },
        "label_grep": "_annotation_flood.tif",
        # Flood masks have three classes (plus -1 = ignore), unlike binary wildfire.
        "num_classes": 3,
        "class_weights": [0.044, 0.487, 0.469],
    },
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
        test_split: str | dict[str, str] = None,
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
        no_data_value: float = -9999.0,
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
                test_split may also be a dict {<name>: <split file>} to evaluate several test
                splits in one run (e.g. the full test set plus the seen-events and held-out-events
                subsets). They are tested in dict order, which must match the task's
                test_dataloaders_names.
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
            self.rgb_indices = rgb_indices or {
                "S2L2A": list(range(12)),
                "S1RTC": [-3, -2],
                "DEM": [-1],
            }
            self.rgb_indices = {
                m: v for m, v in self.rgb_indices.items() if m in self.modalities
            }
        else:
            self.rgb_indices = rgb_indices or {m: None for m in self.modalities}

        self.image_grep = image_grep or {
            "S2L2A": "_S2L2A.zarr.zip",
            "S1RTC": "_S1RTC.zarr.zip",
            "DEM": "_DEM.tif",
        }
        if label_grep is not None:
            self.label_grep = label_grep

        # The disaster type is inferred via an optional argument (disaster_type) passed via kwargs,
        # or if the name of the disaster is in the data root path
        disaster_type = kwargs.get("disaster_type")
        if disaster_type not in datasets_stats.keys():
            for k in datasets_stats.keys():
                if k in str(self.data_root).lower():
                    disaster_type = k
                    break
        if disaster_type is None:
            raise ValueError(
                f"Specify disaster type in data root {self.data_root} or add it as init argument (disaster_type) to the Datamodule. "
                f"Allowed disaster types are {list(datasets_stats.keys())}"
            )

        self.label_grep = datasets_stats[disaster_type]["label_grep"]

        self.timesteps = timesteps or [0, 1, 2, 3]
        self.concat_bands = concat_bands
        self.no_data_value = no_data_value
        self.no_data_replace = no_data_replace if no_data_replace is not None else 0

        self.train_transform = wrap_in_compose_is_list(
            train_transform, image_modalities=self.modalities
        )
        self.val_transform = wrap_in_compose_is_list(
            val_transform, image_modalities=self.modalities
        )
        self.test_transform = wrap_in_compose_is_list(
            test_transform, image_modalities=self.modalities
        )

        # Use stds and means from init args if available, otherwise use the default ones.
        if all([means, stds]):
            pass
        else:
            means = {
                m: datasets_stats[disaster_type]["means"][m] for m in self.modalities
            }
            stds = {
                m: datasets_stats[disaster_type]["stds"][m] for m in self.modalities
            }

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
            # A dict test_split defines multiple test splits {<name>: <split file>},
            # tested in dict order. A plain str is treated as a single "test" split.
            test_splits = (
                self.test_split
                if isinstance(self.test_split, dict)
                else {"test": self.test_split}
            )
            self.test_datasets = [
                ImpactMeshDataset(
                    data_root=self.data_root,
                    split_file=split_file,
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
                for split_file in test_splits.values()
            ]
            # NonGeoDataModule's dataloader factory reads self.test_dataset
            self.test_dataset = self.test_datasets[0]
        if stage in ["predict"]:
            if self.predict_data_root is None:
                logging.warning(
                    f"predict_data_root is not specified, using default data_root {self.data_root}."
                )
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

    def test_dataloader(self):
        """Return one DataLoader per test split (a list if there are several)."""
        loaders = []
        for dataset in self.test_datasets:
            self.test_dataset = dataset  # _dataloader_factory reads self.test_dataset
            loaders.append(self._dataloader_factory("test"))
        return loaders if len(loaders) > 1 else loaders[0]
