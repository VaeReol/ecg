# Original work Copyright (c) Meta Platforms, Inc. and affiliates. <https://github.com/facebookresearch/mae>
# Modified work Copyright 2024 ST-MEM paper authors. <https://github.com/bakqui/ST-MEM>

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Dict, Iterable, Optional, Tuple
import pandas as pd
import torch
import torchmetrics

import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module,
                    criterion: torch.nn.Module,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    loss_scaler,
                    log_writer=None,
                    config: Optional[dict] = None,
                    use_amp: bool = True,
                    ) -> Dict[str, float]:
    model.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = config['accum_iter']
    max_norm = config['max_norm']

    optimizer.zero_grad()

    if log_writer is not None:
        print(f'log_dir: {log_writer.log_dir}')
    all_outputs = []
    all_targets = []
    all_ids = []
    all_fname = []
    for data_iter_step, (samples, targets, ids, fname_new) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, config)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(samples)
            probs = torch.sigmoid(outputs)
            loss = criterion(outputs, targets.unsqueeze(1).float())
            # loss = criterion(outputs, targets)
        if epoch == 8:
            all_outputs.append(probs.detach().cpu().squeeze())  # 去掉多余的维度
            all_targets.append(targets.detach().cpu().squeeze())
            all_ids.append(ids.detach().cpu().squeeze())
            for f in fname_new:
                all_fname.append(f)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss,
                    optimizer,
                    clip_grad=max_norm,
                    parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]['lr']
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((epoch + data_iter_step / len(data_loader)) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
    if epoch == 8:
        all_outputs_tensor = torch.cat(all_outputs).numpy()
        all_targets_tensor = torch.cat(all_targets).numpy()
        all_ids_tensor = torch.cat(all_ids).numpy()
        df = pd.DataFrame({
        'output': all_outputs_tensor,
        'target': all_targets_tensor,
        'id': all_ids_tensor,
        'fname': all_fname
            })
        df.to_csv(f'epoch_{epoch}_results.csv', index=False)
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module,
             criterion: torch.nn.Module,
             data_loader: Iterable,
             device: torch.device,
             metric_fn: torchmetrics.Metric,
             output_act: torch.nn.Module,
             epoch: int,
             target_dtype: torch.dtype = torch.long,
             use_amp: bool = True,
             ) -> Tuple[Dict[str, float], Dict[str, float]]:
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    all_outputs = []
    all_targets = []
    all_ids = []
    all_fname = []
    for samples, targets, ids, fname in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            if samples.ndim == 4:  # batch_size, n_drops, n_channels, n_frames
                logits_list = []
                for i in range(samples.size(1)):
                    logits = model(samples[:, i])
                    logits_list.append(logits)
                logits_list = torch.stack(logits_list, dim=1)
                outputs_list = output_act(logits_list)
                
                logits = logits_list.mean(dim=1)
                probs = torch.sigmoid(logits)
                outputs = outputs_list.mean(dim=1)
            else:
                logits = model(samples)
                outputs = output_act(logits)
                # probs = torch.sigmoid(outputs)
            # loss = criterion(logits, targets)
            loss = criterion(outputs, targets.unsqueeze(1).float())
        if epoch == 8 or 9:
            all_outputs.append(probs.detach().cpu().view(-1))  # 去掉多余的维度
            all_targets.append(targets.detach().cpu().view(-1))
            all_ids.append(ids.detach().cpu().view(-1))
            for f in fname:
                all_fname.append(f)
        outputs = misc.concat_all_gather(outputs)
        targets = misc.concat_all_gather(targets).to(dtype=target_dtype)
        print(f"outputs.shape: {outputs.shape}, targets.shape: {targets.shape}")
        metric_fn.update(outputs, targets.unsqueeze(1).float())
        # metric_fn.update(outputs, targets)
        metric_logger.meters['loss'].update(loss.item(), n=samples.size(0))
    if epoch == 8 or 9:
        all_outputs_tensor = torch.cat(all_outputs).numpy()
        all_targets_tensor = torch.cat(all_targets).numpy()
        all_ids_tensor = torch.cat(all_ids).numpy()
        
        df = pd.DataFrame({
        'output': all_outputs_tensor,
        'target': all_targets_tensor,
        'id': all_ids_tensor,
        'fname': all_fname
            })
        df.to_csv(f'epoch_{epoch}_val_results.csv', index=False)
    metric_logger.synchronize_between_processes()
    valid_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    metrics = metric_fn.compute()
    if isinstance(metrics, dict):  # MetricCollection
        metrics = {k: v.item() for k, v in metrics.items()}
    else:
        metrics = {metric_fn.__class__.__name__: metrics.item()}
    metric_str = "  ".join([f"{k}: {v:.3f}" for k, v in metrics.items()])
    metric_str = f"{metric_str} loss: {metric_logger.loss.global_avg:.3f}"
    print(f"* {metric_str}")
    metric_fn.reset()

    return valid_stats, metrics
