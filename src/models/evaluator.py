import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from models.networks import *
from misc.metric_tool import ConfuseMatrixMeter
from misc.logger_tool import Logger
from utils_ import de_norm
import utils_
from tqdm import tqdm

# Decide which device we want to run on
# torch.cuda.current_device()

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CDEvaluator():
    def __init__(self, args, dataloader):
        self.dataloader = dataloader
        self.n_class = args.n_class
        
        # define Device & Model
        self.device = torch.device("cuda:%s" % args.gpu_ids[0] if torch.cuda.is_available() and len(args.gpu_ids)>0 else "cpu")
        print(f"Evaluation running on device: {self.device}")
        self.net_G = define_G(args=args, gpu_ids=args.gpu_ids)
        self.net_G.to(self.device)

        # Metrics & Loggers
        self.running_metric = ConfuseMatrixMeter(n_class=self.n_class)
        self.checkpoint_dir = args.checkpoint_dir
        self.vis_dir = args.vis_dir
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.vis_dir, exist_ok=True)

        logger_path = os.path.join(args.checkpoint_dir, 'log_eval.txt')
        self.logger = Logger(logger_path)
        self.logger.write_dict_str(args.__dict__)

        # Variables for states
        self.G_pred = None
        self.batch = None
        self.batch_id = 0
        self.epoch_acc = 0
        self.vis_count = 0

        # 🔥 全局累积相关变量
        self.global_fp_map = None
        self.global_fn_map = None

    def _load_checkpoint(self, checkpoint_name='best_ckpt.pt'):
        ckpt_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        
        # 1. 检查文件是否存在
        if os.path.exists(ckpt_path):
            self.logger.write(f'Loading checkpoint: {ckpt_path}...\n')
            
            # 2. 尝试安全加载
            try:
                checkpoint = torch.load(
                    ckpt_path,
                    map_location=self.device,
                    weights_only=True
                )
            # 3. 如果安全加载失败，回退到非安全加载
            except Exception as e:
                print("⚠️ weights_only=True failed, fallback to unsafe load")
                checkpoint = torch.load(
                    ckpt_path,
                    map_location=self.device,
                    weights_only=False
                )
            
            # 4. 加载权重并记录信息 (这部分必须和 try/except 处于同一缩进层级)
            self.net_G.load_state_dict(checkpoint['model_G_state_dict'])
            
            best_val_acc = checkpoint.get('best_val_acc', 0.0)
            best_epoch_id = checkpoint.get('best_epoch_id', 0)
            self.logger.write(f'Eval Historical_best_acc = {best_val_acc:.4f} (at epoch {best_epoch_id})\n\n')
            
        # 5. 如果文件不存在，抛出异常 (else 必须和最上面的 if 对齐)
        else:
            raise FileNotFoundError(f'No such checkpoint: {ckpt_path}')

    def _forward_pass(self, batch):
        self.batch = batch
        img_in1 = batch['A'].to(self.device)
        img_in2 = batch['B'].to(self.device)

        out = self.net_G(img_in1, img_in2)

        # 统一主输出（最高分辨率）
        if isinstance(out, (list, tuple)):
            self.G_pred = out[0]
        else:
            self.G_pred = out

    def _update_metric(self):
        target = self.batch['L'].to(self.device).detach()
        pred = self.G_pred.detach()

        if pred.dim() != 4:
            raise ValueError(f"Prediction shape error: {pred.shape}")

        pred = torch.argmax(pred, dim=1)

        if target.dim() == 4:
            target = target.squeeze(1)

        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: pred {pred.shape}, target {target.shape}")

        current_score = self.running_metric.update_cm(
            pr=pred.cpu().numpy(),
            gt=target.cpu().numpy()
        )
        return current_score

    # =====================================================
    # 🔥 全局错误累积 (FP / FN)
    # =====================================================
    def _accumulate_error(self):
        pred = torch.argmax(self.G_pred.detach(), dim=1)
        gt = self.batch['L'].to(self.device).detach()
        if gt.dim() == 4:
            gt = gt.squeeze(1)

        # 计算当前 batch 的 FP 和 FN (0/1 Tensor)
        fp = ((pred == 1) & (gt == 0)).float()
        fn = ((pred == 0) & (gt == 1)).float()

        # 动态初始化全局 Map (假定所有测试图像大小一致)
        if self.global_fp_map is None:
            _, H, W = fp.shape
            self.global_fp_map = torch.zeros((H, W), device=self.device)
            self.global_fn_map = torch.zeros((H, W), device=self.device)

        # 在 Batch 维度累加 (dim=0)
        self.global_fp_map += fp.sum(dim=0)
        self.global_fn_map += fn.sum(dim=0)

    # =====================================================
    # 局部 Batch 状态可视化 (保存图像)
    # =====================================================
    def _collect_running_batch_states(self, running_acc):
        VIS_INTERVAL = 20          # 每多少batch可视化一次
        MAX_VIS_NUM = 50           # 最多保存多少张

        if (self.batch_id % VIS_INTERVAL == 0) and (self.vis_count < MAX_VIS_NUM):
            vis_input = utils_.make_numpy_grid(de_norm(self.batch['A']))
            vis_input2 = utils_.make_numpy_grid(de_norm(self.batch['B']))

            pred = torch.argmax(self.G_pred, dim=1, keepdim=True).float()
            vis_pred = utils_.make_numpy_grid(pred * 255)

            gt = self.batch['L'].to(self.device)
            if gt.dim() == 3: gt = gt.unsqueeze(1)
            vis_gt = utils_.make_numpy_grid(gt.float().cpu() * 255 if gt.max()<=1 else gt.cpu())

            pred_bin = pred.squeeze(1)
            gt_bin = gt.squeeze(1)
            fp = ((pred_bin == 1) & (gt_bin == 0)).float()
            fn = ((pred_bin == 0) & (gt_bin == 1)).float()

            vis_fp = utils_.make_numpy_grid(fp.unsqueeze(1).cpu() * 255)
            vis_fn = utils_.make_numpy_grid(fn.unsqueeze(1).cpu() * 255)

            # 拼接 6 图
            vis = np.concatenate([vis_input, vis_input2, vis_pred, vis_gt, vis_fp, vis_fn], axis=0)
            vis = np.clip(vis, 0.0, 1.0)

            file_name = os.path.join(self.vis_dir, f'eval_batch_{self.batch_id}_F1_{running_acc:.4f}.jpg')
            plt.imsave(file_name, vis)
            self.vis_count += 1

    # =====================================================
    # 🔥 分布分析与 Heatmap 生成
    # =====================================================
    def _generate_heatmap(self):
        self.logger.write(">>> Generating global error heatmap & Distribution Analysis...\n")

        fp_np = self.global_fp_map.cpu().numpy()
        fn_np = self.global_fn_map.cpu().numpy()

        # --- 分布分析统计 ---
        total_images = len(self.dataloader.dataset)
        self.logger.write(f"Total evaluated images: {total_images}\n")
        self.logger.write(f"[FP Distribution] Max accum: {fp_np.max()}, Mean: {fp_np.mean():.4f}, Std: {fp_np.std():.4f}\n")
        self.logger.write(f"[FN Distribution] Max accum: {fn_np.max()}, Mean: {fn_np.mean():.4f}, Std: {fn_np.std():.4f}\n")

        # 为了更明显的视觉效果，可以除以测试集总数得到概率，或直接做Max归一化
        fp_norm = fp_np / (fp_np.max() + 1e-8)
        fn_norm = fn_np / (fn_np.max() + 1e-8)

        # ===== FP heatmap =====
        plt.figure(figsize=(8, 6))
        plt.imshow(fp_norm, cmap='jet') # jet是经典的蓝-黄-红渐变
        plt.colorbar(label='Normalized Accumulation')
        plt.title(f'False Positive (FP) Heatmap (Max: {fp_np.max():.0f})')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(self.vis_dir, 'global_fp_heatmap.png'), dpi=300)
        plt.close()

        # ===== FN heatmap =====
        plt.figure(figsize=(8, 6))
        plt.imshow(fn_norm, cmap='jet')
        plt.colorbar(label='Normalized Accumulation')
        plt.title(f'False Negative (FN) Heatmap (Max: {fn_np.max():.0f})')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(self.vis_dir, 'global_fn_heatmap.png'), dpi=300)
        plt.close()

    def _collect_epoch_states(self):
        scores_dict = self.running_metric.get_scores()
        self.epoch_acc = scores_dict['mf1']

        message = '==================== Final Eval Results ====================\n'
        for k, v in scores_dict.items():
            message += f'{k}: {v:.5f}\n'
        self.logger.write(message + '\n')

        # 🔥 生成 heatmap 和统计信息
        self._generate_heatmap()

    # =====================================================
    # Eval 主入口
    # =====================================================
    def eval_models(self, checkpoint_name='best_ckpt.pt'):
        self._load_checkpoint(checkpoint_name)
        self.logger.write('Begin evaluation...\n')

        self.running_metric.clear()
        self.net_G.eval()
        
        # 重置全局累加器
        self.global_fp_map = None
        self.global_fn_map = None
        self.vis_count = 0

        # 使用 tqdm 包装 dataloader 提供优美的进度条
        eval_loader = tqdm(self.dataloader, desc="Evaluating", leave=False)

        for self.batch_id, batch in enumerate(eval_loader):
            with torch.no_grad():
                self._forward_pass(batch)
                running_acc = self._update_metric()

                # 🔥 累积全局 error (FP/FN)
                self._accumulate_error()

                # 可视化局部 batch
                self._collect_running_batch_states(running_acc)

                # 更新 tqdm 的后缀信息
                eval_loader.set_postfix(mF1=f"{running_acc:.4f}")

                # 按照一定频率打印到log
                if self.batch_id % 50 == 0:
                    self.logger.write(f'Eval [{self.batch_id}/{len(self.dataloader)}] running_mF1: {running_acc:.5f}\n')

        # Eval 结束，统计与出图
        self._collect_epoch_states()
        self.logger.write('Evaluation Finished.\n')

