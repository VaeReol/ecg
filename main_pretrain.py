# Original work Copyright (c) Meta Platforms, Inc. and affiliates. <https://github.com/facebookresearch/mae>
# Modified work Copyright 2024 ST-MEM paper authors. <https://github.com/bakqui/ST-MEM>

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

import argparse
import datetime
import json
import math
import os
import time
import torch.nn as nn
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import yaml
from torch.utils.tensorboard import SummaryWriter
from new_model import LitEEGPT
import util.misc as misc
from engine_pretrain import train_one_epoch
from util.dataset import build_dataset, get_dataloader
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.optimizer import get_optimizer_from_config
from torch.utils.data import ConcatDataset

def parse() -> dict:
    parser = argparse.ArgumentParser('ECG self-supervised pre-training')

    parser.add_argument('--config_path',
                        default='./configs/pretrain/st_mem_vit_base_12lead.yaml',
                        type=str,
                        metavar='FILE',
                        help='YAML config file path')
    parser.add_argument('--output_dir',
                        default="",
                        type=str,
                        metavar='DIR',
                        help='path where to save')
    parser.add_argument('--exp_name',
                        default="",
                        type=str,
                        help='experiment name')
    parser.add_argument('--resume',
                        default="",
                        type=str,
                        metavar='PATH',
                        help='resume from checkpoint')
    parser.add_argument('--start_epoch',
                        default=0,
                        type=int,
                        metavar='N',
                        help='start epoch')

    args = parser.parse_args()
    with open(os.path.realpath(args.config_path), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for k, v in vars(args).items():
        if v:
            config[k] = v

    return config
def get_config(embed_dim=512, embed_num=1, depth=[8,6,6], num_heads=4): # 23977441
    
    models_configs = {
            'encoder': {
                    'embed_dim': embed_dim,
                    'embed_num': embed_num,
                    'depth': depth[0],
                    'num_heads': num_heads,
                },
            'predictor': {
                    'embed_dim': embed_dim,
                    'embed_num': embed_num,
                    'predictor_embed_dim': embed_dim,
                    'depth': depth[1],
                    'num_heads': num_heads,
                },
            'reconstructor': {
                    'embed_dim': embed_dim,
                    'embed_num': embed_num,
                    'reconstructor_embed_dim': embed_dim,
                    'depth': depth[2],
                    'num_heads': num_heads,
                },
    }
    return models_configs

def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main(config):
    misc.init_distributed_mode(config['ddp'])

    print(f'job dir: {os.path.dirname(os.path.realpath(__file__))}')
    print(yaml.dump(config, default_flow_style=False, sort_keys=False))

    device = torch.device(config['device'])

    # fix the seed for reproducibility
    seed = config['seed'] + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    # config2 = {
    #     'index_dir': '/ssd1/stmem',
    #     'fs': 250,
    #     'filename_col': 'FILE_NAME',
    #     'fs_col': 'SAMPLE_RATE',
    #     'lead': '12lead',
    #     'ecg_dir': '/ssd1/stmem/train_data',
    #     'train_csv': 'index.csv'

    # }
    # ECG dataset
    # dataset_train_mimic = build_dataset(config['dataset_mimic'], split='train')
    dataset_train_code15 =  build_dataset(config['dataset_code15'], split='train')
    dataset_train_ningbo =  build_dataset(config['dataset_ningbo'], split='train')
    # 预训练数据集构建，包括：重采样、滤波、归一化、cutoff
    print(f'ningbo的数量为{len(dataset_train_ningbo)}')
    print(f'code15的数量为{len(dataset_train_code15)}')
    dataset_train = ConcatDataset([dataset_train_ningbo, dataset_train_code15])
    data_loader_train = get_dataloader(dataset_train,
                                       is_distributed=config['ddp']['distributed'],
                                       mode='train',
                                       **config['dataloader'])

    if misc.is_main_process() and config['output_dir']:
        output_dir = os.path.join(config['output_dir'], config['exp_name'])
        os.makedirs(output_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=output_dir)
    else:
        output_dir = None
        log_writer = None
    
    # define the model
    model_configs = get_config()
    model = LitEEGPT(models_configs=model_configs)
    total_params = count_parameters(model)
    model.to(device)

    model_without_ddp = model
    print(f"Model = {model_without_ddp}")

    eff_batch_size = config['dataloader']['batch_size'] * config['train']['accum_iter'] * misc.get_world_size()

    if config['train']['lr'] is None:
        config['train']['lr'] = config['train']['blr'] * eff_batch_size / 256

    print(f"base lr: {config['train']['lr'] * 256 / eff_batch_size}")
    print(f"actual lr: {config['train']['lr']}")
    print(f"accumulate grad iterations: {config['train']['accum_iter']}")
    print(f"effective batch size: {eff_batch_size}")

    if config['ddp']['distributed']:
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[config['ddp']['gpu']])
        model_without_ddp = model.module

    optimizer = get_optimizer_from_config(config['train'], model_without_ddp)
    print(optimizer)
    loss_scaler = NativeScaler()
    max_epochs = config['train']['epochs']
    devices_num = config['num_device']
    steps_per_epoch = math.ceil(len(data_loader_train)/devices_num)
    misc.load_model(config, model_without_ddp, optimizer, loss_scaler)
    print(f"Start training for {config['train']['epochs']} epochs")
    print(f"Total trainable parameters: {total_params}")
    start_time = time.time()
    for epoch in range(config['start_epoch'], config['train']['epochs']):
        if config['ddp']['distributed']:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(model,
                                      data_loader_train,
                                      optimizer,
                                      device,
                                      epoch,
                                      loss_scaler,
                                      log_writer,
                                      max_epochs,
                                      steps_per_epoch,
                                      config['train'])
        if output_dir and (epoch % 20 == 0 or epoch + 1 == config['train']['epochs']):
            misc.save_model(config,
                            os.path.join(output_dir, f'checkpoint-{epoch}.pth'),
                            epoch,
                            model_without_ddp,
                            optimizer,
                            loss_scaler)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch,
                     }

        if output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(output_dir, 'log.txt'), 'a', encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + '\n')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'Training time {total_time_str}')

    # extract encoder
    encoder = model_without_ddp.encoder
    if output_dir:
        misc.save_model(config,
                        os.path.join(output_dir, 'encoder.pth'),
                        epoch,
                        encoder)


if __name__ == "__main__":
    config = parse()
    main(config)
