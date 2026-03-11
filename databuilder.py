import torch
from torch.utils.data import DataLoader

from data.dataset import H5PatchDataset, H5CubeDataset


class DataLoaderBuilder:

    def __init__(
        self,
        *,
        dataset_type="patch",
        batch_size=1,
        num_workers=4,
        pin_memory=True,
    ):

        self.dataset_type = dataset_type
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    # ------------------------------------------------
    # DATASET
    # ------------------------------------------------

    def build_dataset(self, h5_file):

        if self.dataset_type == "patch":
            DatasetClass = H5PatchDataset

        elif self.dataset_type == "cube":
            DatasetClass = H5CubeDataset

        else:
            raise ValueError("dataset_type must be 'patch' or 'cube'")

        dataset = DatasetClass(h5_file)

        return dataset

    # ------------------------------------------------
    # DATALOADER
    # ------------------------------------------------

    def build_dataloader(self, dataset, shuffle):

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

        return loader

    # ------------------------------------------------
    # TRAIN / VAL
    # ------------------------------------------------

    def build(self, train_h5, val_h5=None):

        train_dataset = self.build_dataset(train_h5)

        train_loader = self.build_dataloader(
            train_dataset,
            shuffle=True,
        )

        val_loader = None

        if val_h5 is not None:

            val_dataset = self.build_dataset(val_h5)

            val_loader = self.build_dataloader(
                val_dataset,
                shuffle=False,
            )

        return train_loader, val_loader
