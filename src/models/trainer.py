import numpy as np
import matplotlib.pyplot as plt
import os
import time

import utils_
from models.networks import *

import torch
import torch.optim as optim
from tqdm import tqdm
from misc.metric_tool import ConfuseMatrixMeter
from models.losses import cross_entropy, FocalLoss,BoundaryLoss,Boundary_ce_loss,Focal_Dice_BL,FocalLoss_with_dice,BCEDiceLoss


from misc.logger_tool import Logger, Timer

from utils_ import de_norm

# 定义包装类来处理函数形式的损失函数
class BCEDiceLossClass:
    def __init__(self):
        pass

    def __call__(self, inputs, targets):
        return BCEDiceLoss(inputs, targets)

class BoundaryLossClass:
    def __init__(self):
        pass

    def __call__(self, pred, targets):
        return BoundaryLoss(pred, targets)


class CDTrainer():

    def __init__(self, args, dataloaders):

        self.dataloaders = dataloaders

        self.n_class = args.n_class
        # define G
        self.net_G = define_G(args=args, gpu_ids=args.gpu_ids)

        if torch.cuda.is_available() and len(args.gpu_ids) > 0:
            self.device = torch.device(f"cuda:{args.gpu_ids[0]}")
        else:
            self.device = torch.device("cpu")
        print(self.device)
        self.use_amp = bool(getattr(args, 'use_amp', False) and self.device.type == 'cuda')
        self.amp_dtype = getattr(args, 'amp_dtype', 'fp16')
        self.cache_clear_interval = max(0, int(getattr(args, 'cache_clear_interval', 10) or 0))
        self.log_memory = bool(getattr(args, 'log_memory', True))
        self.scaler = utils_.build_grad_scaler(self.device, self.use_amp)

        # Learning rate and Beta1 for Adam optimizers
        self.lr = args.lr

        # define optimizers
        if args.optimizer == "sgd":
            self.optimizer_G = optim.SGD(self.net_G.parameters(), lr=self.lr,
                                         momentum=0.9,
                                         weight_decay=5e-4)
        elif args.optimizer == "adam":
            self.optimizer_G = optim.Adam(self.net_G.parameters(), lr=self.lr,
                                          weight_decay=0)
        elif args.optimizer == "adamw":
            self.optimizer_G = optim.AdamW(self.net_G.parameters(), lr=self.lr,
                                           betas=(0.9, 0.999), weight_decay=0.01)

       # define lr schedulers  ✅改为 CosineAnnealingLR
        self.exp_lr_scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(
        self.optimizer_G,
        T_max=args.max_epochs,
        eta_min=1e-6
        )

        self.running_metric = ConfuseMatrixMeter(n_class=2)

        # define logger file
        logger_path = os.path.join(args.checkpoint_dir, 'log.txt')
        self.logger = Logger(logger_path)
        self.logger.write_dict_str(args.__dict__)
        self.logger.write(
            f"AMP enabled: {self.use_amp} | amp_dtype: {self.amp_dtype} | "
            f"cache_clear_interval: {self.cache_clear_interval} | log_memory: {self.log_memory}\n"
        )
        # define timer
        self.timer = Timer()
        self.batch_size = args.batch_size

        #  training log
        self.epoch_acc = 0
        self.best_val_acc = 0.0
        self.best_epoch_id = 0
        self.epoch_to_start = 0
        self.max_num_epochs = args.max_epochs

        self.global_step = 0
        self.steps_per_epoch = len(dataloaders['train'])
        self.total_steps = (self.max_num_epochs - self.epoch_to_start)*self.steps_per_epoch

        self.G_pred = None
        self.pred_vis = None
        self.batch = None
        self.G_loss = None
        self.G_loss_id = 0 #表明是哪一个损失函数
        self.is_training = False
        self.batch_id = 0
        self.epoch_id = 0
        self.checkpoint_dir = args.checkpoint_dir
        self.vis_dir = args.vis_dir

       # ===== Loss 修改（组合 Loss）=====
        self.bce_dice_loss = BCEDiceLossClass()
        self.boundary_loss = BoundaryLossClass()
        # 新增：从 args 读取可调 loss 参数
        self.boundary_weight = getattr(args, 'boundary_weight', 0.5)
        self.loss_weights = getattr(args, 'loss_weights', [1.0, 0.8, 0.6, 0.4])

        if len(self.loss_weights) < 4:
            raise ValueError(f"loss_weights 长度不足，当前为 {len(self.loss_weights)}，至少需要 4 个")
        # else:
        #     raise NotImplemented(args.loss)

        self.VAL_ACC = np.array([], np.float32)
        if os.path.exists(os.path.join(self.checkpoint_dir, 'val_acc.npy')):
            self.VAL_ACC = np.load(os.path.join(self.checkpoint_dir, 'val_acc.npy'))
        self.TRAIN_ACC = np.array([], np.float32)
        if os.path.exists(os.path.join(self.checkpoint_dir, 'train_acc.npy')):
            self.TRAIN_ACC = np.load(os.path.join(self.checkpoint_dir, 'train_acc.npy'))

        # check and create model dir
        if os.path.exists(self.checkpoint_dir) is False:
            os.mkdir(self.checkpoint_dir)
        if os.path.exists(self.vis_dir) is False:
            os.mkdir(self.vis_dir)


    def _load_checkpoint(self, ckpt_name='last_ckpt.pt'):

        if os.path.exists(os.path.join(self.checkpoint_dir, ckpt_name)):
            self.logger.write('loading last checkpoint...\n')
            # load the entire checkpoint
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, ckpt_name),
                                    map_location=self.device)
            # update net_G states
            self.net_G.load_state_dict(checkpoint['model_G_state_dict'])

            self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
            self.exp_lr_scheduler_G.load_state_dict(
                checkpoint['exp_lr_scheduler_G_state_dict'])
            if self.use_amp and 'scaler_state_dict' in checkpoint:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

            self.net_G.to(self.device)

            # update some other states
            self.epoch_to_start = checkpoint['epoch_id'] + 1
            self.best_val_acc = checkpoint['best_val_acc']
            self.best_epoch_id = checkpoint['best_epoch_id']

            self.total_steps = (self.max_num_epochs - self.epoch_to_start)*self.steps_per_epoch

            self.logger.write('Epoch_to_start = %d, Historical_best_acc = %.4f (at epoch %d)\n' %
                  (self.epoch_to_start, self.best_val_acc, self.best_epoch_id))
            self.logger.write('\n')

        else:
            print('training from scratch...')

    def _timer_update(self):
        self.global_step = (self.epoch_id-self.epoch_to_start) * self.steps_per_epoch + self.batch_id

        self.timer.update_progress((self.global_step + 1) / self.total_steps)
        est = self.timer.estimated_remaining()
        imps = (self.global_step + 1) * self.batch_size / self.timer.get_stage_elapsed()
        return imps, est

    def _visualize_pred(self, args):
        pred = torch.argmax(self.G_pred, dim=1, keepdim=True)
        pred_vis = pred * 255
        return pred_vis

    def _save_checkpoint(self, ckpt_name):
        payload = {
            'epoch_id': self.epoch_id,
            'best_val_acc': self.best_val_acc,
            'best_epoch_id': self.best_epoch_id,
            'model_G_state_dict': self.net_G.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'exp_lr_scheduler_G_state_dict': self.exp_lr_scheduler_G.state_dict(),
        }
        if self.use_amp:
            payload['scaler_state_dict'] = self.scaler.state_dict()
        torch.save(payload, os.path.join(self.checkpoint_dir, ckpt_name))

    def _update_lr_schedulers(self):
        self.exp_lr_scheduler_G.step()

    def _update_metric(self, args):
        target = self.batch['L'].to(self.device).detach()

        G_pred = self.G_pred.detach()   # 已经是主输出

        G_pred = torch.argmax(G_pred, dim=1)

        current_score = self.running_metric.update_cm(
            pr=G_pred.cpu().numpy(),
            gt=target.cpu().numpy()
        )
        return current_score

    def _collect_running_batch_states(self, args):
        # 1. 更新指标 (只用到 self.G_pred 最高分辨率输出)
        running_acc = self._update_metric(args)

        # 2. 验证集(Eval)由于没有过 _backward_G，在这里补算一下 Loss 仅用于日志记录
        if not self.is_training:
            gt = self.batch['L'].to(self.device).long()
            with utils_.build_autocast_context(self.device, self.use_amp, self.amp_dtype):
                self.G_loss = self._compute_loss(self.G_pred_raw, gt)

        # ====== 下面的乱七八糟的重新计算 Loss 的代码全删掉！======
        # 记录日志信息
        m = len(self.dataloaders['train'])
        if self.is_training is False:
            m = len(self.dataloaders['val'])

        imps, est = self._timer_update()
        if np.mod(self.batch_id, 100) == 1:
            message = 'Is_training: %s. [%d,%d][%d,%d], imps: %.2f, est: %.2fh, G_loss: %.5f, running_mf1: %.5f\n' %\
                      (self.is_training, self.epoch_id, self.max_num_epochs-1, self.batch_id, m,
                     imps*self.batch_size, est,
                     self.G_loss.item(), running_acc) # 确保用 .item()
            self.logger.write(message)

        if np.mod(self.batch_id, 500) == 1:
            vis_input = utils_.make_numpy_grid(de_norm(self.batch['A']))
            vis_input2 = utils_.make_numpy_grid(de_norm(self.batch['B']))
            vis_pred = utils_.make_numpy_grid(self._visualize_pred(args))
            vis_gt = utils_.make_numpy_grid(self.batch['L'])
            vis = np.concatenate([vis_input, vis_input2, vis_pred, vis_gt], axis=0)
            vis = np.clip(vis, a_min=0.0, a_max=1.0)
            file_name = os.path.join(
                self.vis_dir, 'istrain_'+str(self.is_training)+'_'+
                              str(self.epoch_id)+'_'+str(self.batch_id)+'.jpg')
            # plt.imsave(file_name, vis)

    def _collect_epoch_states(self):
        scores = self.running_metric.get_scores()
        self.epoch_acc = scores['mf1']
        self.logger.write('Is_training: %s. Epoch %d / %d, epoch_mF1= %.5f\n' %
              (self.is_training, self.epoch_id, self.max_num_epochs-1, self.epoch_acc))
        message = ''
        for k, v in scores.items():
            message += '%s: %.5f ' % (k, v)
        self.logger.write(message+'\n')
        self.logger.write('\n')

    def _update_checkpoints(self):

        # save current model
        self._save_checkpoint(ckpt_name='last_ckpt.pt')
        self.logger.write('Lastest model updated. Epoch_acc=%.4f, Historical_best_acc=%.4f (at epoch %d)\n'
              % (self.epoch_acc, self.best_val_acc, self.best_epoch_id))
        self.logger.write('\n')

        # update the best model (based on eval acc)
        if self.epoch_acc > self.best_val_acc:
            self.best_val_acc = self.epoch_acc
            self.best_epoch_id = self.epoch_id
            self._save_checkpoint(ckpt_name='best_ckpt.pt')
            self.logger.write('*' * 10 + 'Best model updated!\n')
            self.logger.write('\n')

    def _update_training_acc_curve(self):
        # update train acc curve
        self.TRAIN_ACC = np.append(self.TRAIN_ACC, [self.epoch_acc])
        np.save(os.path.join(self.checkpoint_dir, 'train_acc.npy'), self.TRAIN_ACC)

    def _update_val_acc_curve(self):
        # update val acc curve
        self.VAL_ACC = np.append(self.VAL_ACC, [self.epoch_acc])
        np.save(os.path.join(self.checkpoint_dir, 'val_acc.npy'), self.VAL_ACC)

    def _clear_cache(self, force_cuda=False, gc_collect=False):
        self.running_metric.clear()
        if force_cuda:
            utils_.maybe_clear_cuda_cache(force=True, gc_collect=gc_collect)

    def _reset_memory_peak(self):
        if self.device.type == 'cuda':
            utils_.reset_cuda_peak_memory_stats(self.device)

    def _log_memory_state(self, tag):
        if self.log_memory and self.device.type == 'cuda':
            self.logger.write(f"[cuda_memory] {tag}: {utils_.format_cuda_memory_stats(self.device)}\n")

    def _compute_loss(self, preds, gt):
        loss = 0.0

        if gt.dim() == 3:
            gt_4d = gt.unsqueeze(1).float()
        else:
            gt_4d = gt.float()

        if isinstance(preds, (list, tuple)):  # 多尺度深度监督
            weights = self.loss_weights

            for i, pred in enumerate(preds):
                # 将 GT 插值到当前特征图大小
                gt_resized_4d = F.interpolate(gt_4d, size=pred.shape[2:], mode='nearest')
                gt_resized = gt_resized_4d.squeeze(1).long()

                # 当前尺度 loss
                curr_loss = self.bce_dice_loss(pred, gt_resized) + \
                            self.boundary_weight * self.boundary_loss(pred, gt_resized)

                # 加权累加
                if i < len(weights):
                    loss += weights[i] * curr_loss
                else:
                    loss += curr_loss

            loss = loss / len(preds)

        else:  # 单输出（Eval阶段）
            gt_resized = gt_4d.squeeze(1).long()
            loss = self.bce_dice_loss(preds, gt_resized) + \
                   self.boundary_weight * self.boundary_loss(preds, gt_resized)

        return loss
    def _forward_pass(self, args, batch):
        self.batch = batch
        img_in1 = batch['A'].to(self.device)
        img_in2 = batch['B'].to(self.device)

        # 拿到模型输出
        with utils_.build_autocast_context(self.device, self.use_amp, self.amp_dtype):
            out = self.net_G(img_in1, img_in2)
        
        # 保存完整的输出用于计算 Loss (如果是多尺度，out 是 tuple)
        self.G_pred_raw = out  

        # 仅取出最高分辨率的主输出，用于计算评价指标 (Acc, F1 等)
        if isinstance(out, tuple):
            self.G_pred = out[0]
        else:
            self.G_pred = out

    def _backward_G(self, args):
        gt = self.batch['L'].to(self.device).long()
        
        # 1. 调用统一的 loss 计算函数
        with utils_.build_autocast_context(self.device, self.use_amp, self.amp_dtype):
            self.G_loss = self._compute_loss(self.G_pred_raw, gt)
        
        # 2. 反向传播
        if self.use_amp:
            self.scaler.scale(self.G_loss).backward()
        else:
            self.G_loss.backward()


    def train_models(self,args):

        self._load_checkpoint()

        # loop over the dataset multiple times
        for self.epoch_id in range(self.epoch_to_start, self.max_num_epochs):
            epoch_started = time.time()

            ################## train #################
            ##########################################
            self._clear_cache(force_cuda=True, gc_collect=True)
            self._reset_memory_peak()
            self.is_training = True
            self.net_G.train()  # Set model to training mode
            self._log_memory_state(f'epoch_{self.epoch_id}_train_start')
            # Iterate over data.
            total = len(self.dataloaders['train'])
            self.logger.write('lr: %0.7f\n' % self.optimizer_G.param_groups[0]['lr'])
            for self.batch_id, batch in tqdm(enumerate(self.dataloaders['train'], 0),total=total):
                self._forward_pass(args,batch)
                # update G
                self.optimizer_G.zero_grad(set_to_none=True)
                # for name, parms in self.net_G.named_parameters():
                #     print('-->name:', name)
                #     print('-->para:', parms)
                #     print('-->grad_requirs:', parms.requires_grad)
                #     print('-->grad_value:', parms.grad)
                #     print("===")

                self._backward_G(args)
                if self.use_amp:
                    self.scaler.step(self.optimizer_G)
                    self.scaler.update()
                else:
                    self.optimizer_G.step()

                self._collect_running_batch_states(args)
                self._timer_update()
                utils_.maybe_clear_cuda_cache(
                    step=self.batch_id,
                    interval=self.cache_clear_interval,
                    gc_collect=False,
                )

            self._collect_epoch_states()
            self._update_training_acc_curve()
            self._update_lr_schedulers()
            self._log_memory_state(f'epoch_{self.epoch_id}_train_end')
            utils_.maybe_clear_cuda_cache(force=True, gc_collect=True)


            ################## Eval ##################
            ##########################################
            self.logger.write('Begin evaluation...\n')
            self._clear_cache(force_cuda=True, gc_collect=True)
            self._reset_memory_peak()
            self.is_training = False
            self.net_G.eval()
            self._log_memory_state(f'epoch_{self.epoch_id}_val_start')

            # Iterate over data.
            for self.batch_id, batch in enumerate(self.dataloaders['val'], 0):
                with torch.no_grad():
                    self._forward_pass(args,batch)
                self._collect_running_batch_states(args)
            self._collect_epoch_states()
            self._log_memory_state(f'epoch_{self.epoch_id}_val_end')
            utils_.maybe_clear_cuda_cache(force=True, gc_collect=True)

            ########### Update_Checkpoints ###########
            ##########################################
            self._update_val_acc_curve()
            self._update_checkpoints()

            train_mf1 = self.TRAIN_ACC[-1] if len(self.TRAIN_ACC) > 0 else self.epoch_acc
            val_mf1 = self.VAL_ACC[-1] if len(self.VAL_ACC) > 0 else self.epoch_acc
            epoch_minutes = (time.time() - epoch_started) / 60.0
            remaining_after_this = self.max_num_epochs - self.epoch_id - 1
            rough_remaining_hours = epoch_minutes * remaining_after_this / 60.0
            self.logger.write(
                utils_.format_epoch_summary(
                    epoch_id=self.epoch_id,
                    max_epochs=self.max_num_epochs,
                    train_mf1=train_mf1,
                    val_mf1=val_mf1,
                    epoch_minutes=epoch_minutes,
                    remaining_hours=rough_remaining_hours,
                    best_epoch=self.best_epoch_id,
                    best_val_mf1=self.best_val_acc,
                    amp_enabled=self.use_amp,
                    amp_dtype=self.amp_dtype,
                    device=self.device,
                ) + '\n\n'
            )

