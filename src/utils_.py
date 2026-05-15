import gc
from contextlib import nullcontext

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import utils

import data_config
from datasets.CD_dataset import CDDataset
import argparse

def build_dataloader(
    dataset,
    *,
    batch_size,
    shuffle,
    num_workers,
    pin_memory=None,
):
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def rebuild_dataloader(loader, *, num_workers, shuffle=None, pin_memory=None):
    if shuffle is None:
        sampler_name = type(loader.sampler).__name__.lower()
        shuffle = "random" in sampler_name

    return build_dataloader(
        loader.dataset,
        batch_size=loader.batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def get_loader(data_name, img_size=256, batch_size=8, split='test',
               is_train=False, dataset='CDDataset', num_workers=4, pin_memory=None,
               data_root=None):
    dataConfig = data_config.DataConfig().get_data_config(data_name, root_dir=data_root)
    root_dir = dataConfig.root_dir
    label_transform = dataConfig.label_transform

    if dataset == 'CDDataset':
        data_set = CDDataset(root_dir=root_dir, split=split,
                                 img_size=img_size, is_train=is_train,
                                 label_transform=label_transform)
    else:
        raise NotImplementedError(
            'Wrong dataset name %s (choose one from [CDDataset])'
            % dataset)

    shuffle = is_train
    dataloader = build_dataloader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


    return dataloader


def get_loaders(args):

    data_name = args.data_name
    data_root = getattr(args, 'data_root', None)
    dataConfig = data_config.DataConfig().get_data_config(data_name, root_dir=data_root)
    root_dir = dataConfig.root_dir
    label_transform = dataConfig.label_transform
    split = args.split
    split_val = 'val'
    if hasattr(args, 'split_val'):
        split_val = args.split_val
    if args.dataset == 'CDDataset':
        training_set = CDDataset(root_dir=root_dir, split=split,
                                 img_size=args.img_size,is_train=True,
                                 label_transform=label_transform)
        val_set = CDDataset(root_dir=root_dir, split=split_val,
                                 img_size=args.img_size,is_train=False,
                                 label_transform=label_transform)
    else:
        raise NotImplementedError(
            'Wrong dataset name %s (choose one from [CDDataset,])'
            % args.dataset)

    datasets = {'train': training_set, 'val': val_set}
    dataloaders = {
        'train': build_dataloader(
            datasets['train'],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        ),
        'val': build_dataloader(
            datasets['val'],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        ),
    }

    return dataloaders


def make_numpy_grid(tensor_data, pad_value=0,padding=0):
    tensor_data = tensor_data.detach()
    vis = utils.make_grid(tensor_data, pad_value=pad_value,padding=padding)
    vis = np.array(vis.cpu()).transpose((1,2,0))
    if vis.shape[2] == 1:
        vis = np.stack([vis, vis, vis], axis=-1)
    return vis


def de_norm(tensor_data):
    return tensor_data * 0.5 + 0.5


def get_device(args):
    if isinstance(args.gpu_ids, str):
        str_ids = args.gpu_ids.split(',')
        args.gpu_ids = [int(id) for id in str_ids if int(id) >= 0]
    elif isinstance(args.gpu_ids, list):
        # 已经处理过，直接用
        args.gpu_ids = [int(id) for id in args.gpu_ids if int(id) >= 0]

    if len(args.gpu_ids) > 0:
        torch.cuda.set_device(args.gpu_ids[0])


def build_autocast_context(device, enabled=False, amp_dtype='fp16'):
    if not enabled or device.type != 'cuda':
        return nullcontext()

    dtype = torch.float16
    if str(amp_dtype).lower() == 'bf16':
        dtype = torch.bfloat16

    try:
        return torch.amp.autocast(device_type='cuda', dtype=dtype)
    except AttributeError:
        return torch.cuda.amp.autocast(dtype=dtype)


def build_grad_scaler(device, enabled=False):
    amp_enabled = bool(enabled and device.type == 'cuda')
    try:
        return torch.amp.GradScaler('cuda', enabled=amp_enabled)
    except AttributeError:
        return torch.cuda.amp.GradScaler(enabled=amp_enabled)


def maybe_clear_cuda_cache(step=None, interval=0, force=False, gc_collect=False):
    if not torch.cuda.is_available():
        return False

    should_clear = force
    if not should_clear and interval and interval > 0 and step is not None:
        should_clear = (step + 1) % int(interval) == 0

    if not should_clear:
        return False

    if gc_collect:
        gc.collect()
    torch.cuda.empty_cache()
    return True


def get_cuda_memory_stats(device=None):
    if not torch.cuda.is_available():
        return {
            'allocated_mb': 0.0,
            'reserved_mb': 0.0,
            'max_allocated_mb': 0.0,
            'max_reserved_mb': 0.0,
        }

    if device is None:
        device = torch.device('cuda:0')

    allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
    max_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    max_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
    return {
        'allocated_mb': allocated,
        'reserved_mb': reserved,
        'max_allocated_mb': max_allocated,
        'max_reserved_mb': max_reserved,
    }


def format_cuda_memory_stats(device=None):
    stats = get_cuda_memory_stats(device)
    return (
        f"allocated={stats['allocated_mb']:.1f}MB, "
        f"reserved={stats['reserved_mb']:.1f}MB, "
        f"max_allocated={stats['max_allocated_mb']:.1f}MB, "
        f"max_reserved={stats['max_reserved_mb']:.1f}MB"
    )


def format_cuda_peak_stats(device=None):
    stats = get_cuda_memory_stats(device)
    return (
        f"peak_allocated={stats['max_allocated_mb']:.1f}MB, "
        f"peak_reserved={stats['max_reserved_mb']:.1f}MB"
    )


def format_remaining_time(hours: float) -> str:
    hours = max(float(hours), 0.0)
    if hours < 1:
        return f"{hours * 60:.1f} min"
    return f"{hours:.2f} h"


def format_epoch_summary(
    *,
    epoch_id,
    max_epochs,
    train_mf1,
    val_mf1,
    epoch_minutes,
    remaining_hours,
    best_epoch,
    best_val_mf1,
    amp_enabled,
    amp_dtype,
    device=None,
):
    return (
        f"[epoch_summary] "
        f"epoch={epoch_id}/{max_epochs - 1} | "
        f"train_mF1={train_mf1:.5f} | "
        f"val_mF1={val_mf1:.5f} | "
        f"epoch_time={epoch_minutes:.2f} min | "
        f"remaining={format_remaining_time(remaining_hours)} | "
        f"AMP={'ON' if amp_enabled else 'OFF'}"
        f"{'(' + str(amp_dtype) + ')' if amp_enabled else ''} | "
        f"{format_cuda_peak_stats(device)} | "
        f"best_epoch={best_epoch} | "
        f"best_val_mF1={best_val_mf1:.5f}"
    )


def reset_cuda_peak_memory_stats(device=None):
    if not torch.cuda.is_available():
        return
    if device is None:
        device = torch.device('cuda:0')
    torch.cuda.reset_peak_memory_stats(device)

def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
