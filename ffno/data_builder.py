import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from data.dataset import H5PatchDataset, H5CubeDataset


def is_distributed():
    return dist.is_available() and dist.is_initialized()


class DataLoaderBuilder:

    def __init__(
        self,
        *,
        dataset_type="patch",
        batch_size=1,
        num_workers=4,
        pin_memory=True,
        train_select=1.0,
        train_select_seed=None,
    ):

        self.dataset_type = dataset_type
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_select = train_select
        self.train_select_seed = train_select_seed

    # ------------------------------------------------
    # DATASET
    # ------------------------------------------------

    def build_dataset(self, h5_file, *, train_select=None):

        if self.dataset_type == "patch":
            DatasetClass = H5PatchDataset

        elif self.dataset_type == "cube":
            DatasetClass = H5CubeDataset

        else:
            raise ValueError("dataset_type must be 'patch' or 'cube'")

        if train_select is None:
            train_select = 1.0

        if DatasetClass is H5PatchDataset:
            dataset = DatasetClass(
                h5_file,
                train_select=train_select,
                train_select_seed=self.train_select_seed,
            )
        else:
            if float(train_select) != 1.0:
                raise ValueError("TRAINSELECT is only supported for patch datasets")
            dataset = DatasetClass(h5_file)

        return dataset

    # ------------------------------------------------
    # DATALOADER
    # ------------------------------------------------

    def build_dataloader(self, dataset, shuffle):
        if getattr(dataset, "variable_depth", False) and self.batch_size != 1:
            raise ValueError(
                "Variable native-depth datasets require batch_size=1. "
                "Set BATCH_SIZE = 1 for grouped FFNO training data."
            )

        sampler = None
        persistent_workers = self.num_workers > 0 and not getattr(
            dataset,
            "dynamic_length",
            False,
        )

        if is_distributed():

            sampler = DistributedSampler(
                dataset,
                shuffle=shuffle,
                drop_last=False,
            )

            shuffle = False  # sampler handles shuffling

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=persistent_workers,
        )

        return loader, sampler

    # ------------------------------------------------
    # TRAIN / VAL
    # ------------------------------------------------

    def build(self, train_h5, val_h5=None):

        train_dataset = self.build_dataset(
            train_h5,
            train_select=self.train_select,
        )

        train_loader, train_sampler = self.build_dataloader(
            train_dataset,
            shuffle=True,
        )

        val_loader = None
        val_sampler = None

        if val_h5 is not None:

            val_dataset = self.build_dataset(val_h5)

            val_loader, val_sampler = self.build_dataloader(
                val_dataset,
                shuffle=False,
            )

        return train_loader, val_loader, train_sampler, val_sampler
