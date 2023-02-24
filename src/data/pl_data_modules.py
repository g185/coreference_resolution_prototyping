from functools import partial
from typing import Any, Union, List, Optional, Sequence

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from torch.utils.data import DataLoader, Dataset


class BasePLDataModule(pl.LightningDataModule):

    def __init__(
        self,
        datasets: DictConfig,
        batch_sizes: DictConfig,
        num_workers: DictConfig,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.datasets = datasets
        self.num_workers = num_workers
        self.batch_sizes = batch_sizes
        # data
        self.train_dataset: Optional[Dataset] = None
        self.val_datasets: Optional[Sequence[Dataset]] = None
        self.test_datasets: Optional[Sequence[Dataset]] = None


    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = hydra.utils.instantiate(self.datasets.train[0])
            self.val_datasets = [
                hydra.utils.instantiate(dataset_cfg)
                for dataset_cfg in self.datasets.val
            ]
        if stage == "test" or stage is None:
            self.test_datasets = [
                hydra.utils.instantiate(dataset_cfg)
                for dataset_cfg in self.datasets.test
            ]

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_sizes.train,
            num_workers=self.num_workers.train,
            collate_fn=self.train_dataset.collate_fn
        )

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return [
            DataLoader(
                dataset,
                shuffle=False,
                batch_size=self.batch_sizes.val,
                num_workers=self.num_workers.val,
                collate_fn=dataset.collate_fn
            )
            for dataset in self.val_datasets
        ]

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return [
            DataLoader(
                dataset,
                shuffle=False,
                batch_size=self.batch_sizes.test,
                num_workers=self.num_workers.test,
                collate_fn=dataset.collate_fn
            )
            for dataset in self.test_datasets
        ]


    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{self.datasets=}, "
            f"{self.num_workers=}, "
            f"{self.batch_sizes=})"
        )
