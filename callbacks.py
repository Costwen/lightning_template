import torch
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
import time
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.utilities import rank_zero_info
from typing import Dict, Any
import os
import torch.distributed as dist

class CUDACallback(Callback):
    # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py
    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        torch.cuda.reset_peak_memory_stats(trainer.strategy.root_device.index)
        torch.cuda.synchronize(trainer.strategy.root_device.index)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        torch.cuda.synchronize(trainer.strategy.root_device.index)
        max_memory = torch.cuda.max_memory_allocated(trainer.strategy.root_device.index) / 2 ** 20
        epoch_time = time.time() - self.start_time

        try:
            max_memory = trainer.training_type_plugin.reduce(max_memory)
            epoch_time = trainer.training_type_plugin.reduce(epoch_time)

            rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
            rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")
        except AttributeError:
            pass

class DataCallback(Callback):
    """
        Use this callback to save and load the dataloader state
    """
    def on_save_checkpoint(self, trainer, pl_module, checkpoint: Dict[str, Any]) -> None:
        # save batch_idx and epoch
        checkpoint["batch_idx"] = pl_module.batch_idx
        checkpoint["epoch"] = trainer.current_epoch
        checkpoint["num_replicas"] = dist.get_world_size() if dist.is_initialized() else 1

    def on_load_checkpoint(self, trainer, pl_module, checkpoint: Dict[str, Any]) -> None:
        # get datamodule
        datamodule = trainer.datamodule
        train_dataloader = datamodule.train_dataloader()
        batch_size = datamodule.batch_size
        num_replicas = checkpoint["num_replicas"]
        train_dataloader.sampler.set_epoch(checkpoint["epoch"])
        train_dataloader.sampler.set_start_index(checkpoint["batch_idx"] * batch_size * num_replicas)

class StepCallback(Callback):
    def on_load_checkpoint(self, trainer, pl_module, checkpoint: Dict[str, Any]) -> None:
        pl_module.step = checkpoint["global_step"]
        # check trainer in test
        if trainer.testing:
            print(f"Testing at step {pl_module.step}")
            save_dir = os.path.join("outputs", pl_module.exp_name, str(pl_module.step))
            if os.path.exists(save_dir):
                print(f"save_dir {save_dir} already exists, skipping")
                return