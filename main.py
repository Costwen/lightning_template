import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from omegaconf import OmegaConf
import argparse
from common import get_instance_from_conf
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import os
import torch
from callbacks import *
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def load_callbacks(conf):
    callbacks = []
    for c in conf.callbacks:
        callback = get_instance_from_conf(c)
        callbacks.append(callback)
    return callbacks

def find_latest_checkpoint(path):
    if path.endswith("ckpt"):
        return path
    # epoch \={} - step \={}.ckpt sort by epoch and step
    files = os.listdir(path)
    if len(files) == 0:
        return None
    files = [f for f in files if f.endswith(".ckpt")]
    files = [f for f in files if "epoch" and "step" in f]
    # sort by epoch and step
    epochs = [int(f.split("epoch=")[-1].split("-")[0]) for f in files]
    steps = [int(f.split("step=")[-1].split(".ckpt")[0]) for f in files]
    files = [f for _, f in sorted(zip(epochs, files))]
    files = [f for _, f in sorted(zip(steps, files))]
    return os.path.join(path, files[-1])
    

def main(args):
    pl.seed_everything(args.seed)
    conf = OmegaConf.load(args.config)

    conf.trainer.devices = int(args.devices)
    conf.trainer.num_nodes = int(args.num_nodes)

    bs, base_lr, ngpu, accumulate = conf.data.params.batch_size, conf.model.lr, conf.trainer.devices, conf.trainer.accumulate_grad_batches

    model = get_instance_from_conf(conf.model)
    data = get_instance_from_conf(conf.data)
    log_dir = conf.log_dir
    
    logger = TensorBoardLogger(log_dir, name=conf.exp_name)
    if args.resume:
        # get dir name
        ckpt_dir = os.path.dirname(args.resume)
        version = ckpt_dir.split("/")[-2]
        
        logger = TensorBoardLogger(log_dir, name=conf.exp_name, version=version)
        latest_checkpoint = find_latest_checkpoint(args.resume)
        
        if latest_checkpoint is None:
            print("No checkpoint found in {}".format(args.resume))
        else:
            conf.trainer.resume_from_checkpoint = latest_checkpoint
        
    if args.scale_lr:
        model.lr = accumulate * bs * base_lr * ngpu
        print(
            "Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
                model.lr, accumulate, ngpu, bs, base_lr))
    else:
        model.lr = base_lr
        print(f"Setting learning rate to {model.lr:.2e}")

    trainer_args = argparse.Namespace(**conf["trainer"])
    callbacks = load_callbacks(conf)

    # logger = WandbLogger(project="light-diffusion", name=conf.exp_name, save_dir="logs")
    trainer = Trainer.from_argparse_args(trainer_args, callbacks=callbacks, logger=logger)
    
    if args.train:
        trainer.fit(model, data)
    else:
        trainer.test(model, data, ckpt_path=latest_checkpoint)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--config', default='', type=str)
    parser.add_argument('--train', default=True, type=str2bool)
    parser.add_argument(
        "--scale_lr",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="scale base-lr by ngpu * batch_size * n_accumulate",
    )
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    main(args)