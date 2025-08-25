import math
import sys, random
from typing import Iterable
import torch
import torch.nn.functional as F
import util.misc as misc
import util.lr_sched as lr_sched

def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    loss_scaler,
                    log_writer=None,
                    max_epochs=800,
                    steps_per_epoch=1,
                    config=None):
    print('开始训练')
    model.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = config['accum_iter']

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    ema = [0.996,1.0]
    # 动量更新器
    momentum_scheduler = (ema[0] + i * (ema[1] - ema[0]) / (steps_per_epoch * max_epochs)
                          for i in range(int(steps_per_epoch * max_epochs) + 1))
    
    for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        # 余弦退火学习率调度
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, config)

        samples = samples.type(torch.FloatTensor)
        print(f'==============={samples.shape}=============')
        samples = samples.to(device, non_blocking=True)
        # print('samples加载完毕')
        with torch.cuda.amp.autocast():
            results = model(samples)
        loss = results.get('loss')
        # print(f"loss: {loss}")
        loss_value = loss.item()
        loss1 = results.get('loss1').item()
        loss2 = results.get('loss2').item()
        loss_amp = results.get('loss_amp').item()
        loss_angle = results.get('loss_angle').item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)
        loss = loss / accum_iter
        loss_scaler(loss,
                    optimizer,
                    parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]['lr']
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        loss_1 = misc.all_reduce_mean(loss1)
        loss_2 = misc.all_reduce_mean(loss2)
        loss_amp = misc.all_reduce_mean(loss_amp)
        loss_angle = misc.all_reduce_mean(loss_angle)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((epoch + data_iter_step / len(data_loader)) * 1000)
            log_writer.add_scalar('pretrain_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('pretrain_loss_1', loss_1, epoch_1000x)
            log_writer.add_scalar('pretrain_loss_2', loss_2, epoch_1000x)
            log_writer.add_scalar('pretrain_loss_amp', loss_amp, epoch_1000x)
            log_writer.add_scalar('pretrain_loss_angle', loss_angle, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

        # Momentum update of target encoder
        with torch.no_grad():
            m = next(momentum_scheduler)
            for param_q, param_k in zip(model.module.encoder.parameters(), model.module.target_encoder.parameters()):
                param_k.data.mul_(m).add_((1.0 - m) * param_q.detach().data)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('Averaged stats:', metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

