import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import torch.distributed as dist
import torch.utils.data as data
from torch.utils.data.dataloader import default_collate
from torch.utils.data.distributed import DistributedSampler
import torch
import math

class CustomDistributedSampler(DistributedSampler):
    def __init__(self, dataset, num_replicas, rank, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False, start_index: int = 0):
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.start_index = start_index

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size
        if self.start_index > 0:
            indices = indices[self.start_index:]
            self.start_index = 0
        indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices)

    def set_start_index(self, start_index):
        self.start_index = start_index
        
def collect_fn(batch):
    # li = [i for i in batch if i["video"] is not None]
    return default_collate(batch)

class DataInterface(pl.LightningDataModule):
    def __init__(self, batch_size, dataset, eval_dataset = None, shuffle=True, num_workers=None):
        super().__init__()
        self.dataset = dataset
        self.eval_dataset = eval_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers if num_workers else 2 * batch_size
        self._train_dataloader = None
        
    def _dataloader(self, train):
        if train:
            dataset = self.dataset
        else:
            if self.eval_dataset is None:
                return None
            dataset = self.eval_dataset
            self.batch_size = self.batch_size
        if dist.is_initialized():
            sampler = CustomDistributedSampler(
                dataset,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=train,
            )
        dataloader = data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=sampler,
            shuffle=sampler is None and self.shuffle is True,
            collate_fn=collect_fn,
        )
        return dataloader

    def train_dataloader(self):
        if self._train_dataloader is None:
            self._train_dataloader = self._dataloader(True)
        return self._train_dataloader

    def val_dataloader(self):
        return self._dataloader(False)

    def test_dataloader(self):
        return self.val_dataloader()