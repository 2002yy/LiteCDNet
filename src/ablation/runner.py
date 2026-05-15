from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

import utils_
from ablation.litecdnet_variants import build_ablation_model
from misc.logger_tool import Logger, Timer
from misc.metric_tool import ConfuseMatrixMeter
from models.losses import BCEDiceLoss, BoundaryLoss
from utils_ import de_norm


@dataclass
class LossWrapper:
    fn: callable

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


def ensure_log_writable(log_path: str) -> None:
    parent = os.path.dirname(log_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    try:
        with open(log_path, mode="a"):
            pass
    except PermissionError as exc:
        raise RuntimeError(
            f"Cannot start because log file is not writable: {log_path}\n"
            "Possible cause: a previous training/evaluation process is still running and occupies this file.\n"
            "Suggested action: stop the old Python process, or change project_name/checkpoint_dir before retrying."
        ) from exc


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def acquire_experiment_lock(checkpoint_dir: str) -> str:
    os.makedirs(checkpoint_dir, exist_ok=True)
    lock_path = os.path.join(checkpoint_dir, "train.lock")
    payload = {
        "pid": os.getpid(),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint_dir": checkpoint_dir,
    }

    if os.path.exists(lock_path):
        owner_pid = -1
        owner_created_at = "unknown"
        try:
            with open(lock_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            owner_pid = int(existing.get("pid", -1))
            owner_created_at = str(existing.get("created_at", "unknown"))
        except Exception:
            existing = {}

        if _pid_is_running(owner_pid):
            raise RuntimeError(
                f"Another training process is already using this checkpoint directory: {checkpoint_dir}\n"
                f"Lock file: {lock_path}\n"
                f"Owner PID: {owner_pid} | created_at: {owner_created_at}\n"
                "Suggested action: stop the old Python process first, or use a different project_name/checkpoint_dir."
            )

        try:
            os.remove(lock_path)
            print(f"[lock] Removed stale lock: {lock_path} (owner PID: {owner_pid})", flush=True)
        except OSError as exc:
            raise RuntimeError(
                f"Found a stale lock file but could not remove it: {lock_path}\n"
                "Suggested action: delete the lock file manually and retry."
            ) from exc

    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(
            f"Another training process is already using this checkpoint directory: {checkpoint_dir}\n"
            f"Lock file: {lock_path}\n"
            "Suggested action: stop the old Python process first, or use a different project_name/checkpoint_dir."
        ) from exc

    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[lock] Acquired training lock: {lock_path} | pid={payload['pid']}", flush=True)
    return lock_path


def release_experiment_lock(lock_path: str) -> None:
    if not lock_path:
        return
    try:
        os.remove(lock_path)
        print(f"[lock] Released training lock: {lock_path}", flush=True)
    except FileNotFoundError:
        pass


class AblationTrainer:
    def __init__(self, args, dataloaders):
        self.args = args
        self.dataloaders = dataloaders
        self.device = torch.device(
            f"cuda:{args.gpu_ids[0]}" if torch.cuda.is_available() and len(args.gpu_ids) > 0 else "cpu"
        )
        self.use_amp = bool(getattr(args, "use_amp", False) and self.device.type == "cuda")
        self.amp_dtype = getattr(args, "amp_dtype", "fp16")
        self.cache_clear_interval = max(0, int(getattr(args, "cache_clear_interval", 10) or 0))
        self.log_memory = bool(getattr(args, "log_memory", True))
        self.scaler = utils_.build_grad_scaler(self.device, self.use_amp)

        self.net_G = build_ablation_model(args).to(self.device)
        if torch.cuda.is_available() and len(args.gpu_ids) > 1:
            self.net_G = torch.nn.DataParallel(self.net_G, args.gpu_ids)

        self.optimizer_G = optim.AdamW(
            self.net_G.parameters(),
            lr=args.lr,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_G, T_max=args.max_epochs, eta_min=1e-6
        )

        self.metric = ConfuseMatrixMeter(n_class=2)
        self.timer = Timer()
        self.steps_per_epoch = len(self.dataloaders["train"])
        self.epoch_to_start = 0
        self.total_steps = max(1, self.steps_per_epoch * args.max_epochs)
        self.global_step = 0
        self.batch_size = args.batch_size
        self.max_num_epochs = args.max_epochs
        self.is_training = False
        self.epoch_id = 0

        self.checkpoint_dir = args.checkpoint_dir
        self.vis_dir = args.vis_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.vis_dir, exist_ok=True)

        self.log_path = os.path.join(self.checkpoint_dir, "log.txt")
        ensure_log_writable(self.log_path)
        self.logger = Logger(self.log_path)
        self.logger.write_dict_str(args.__dict__)
        self.logger.write(
            f"AMP enabled: {self.use_amp} | amp_dtype: {self.amp_dtype} | "
            f"cache_clear_interval: {self.cache_clear_interval} | log_memory: {self.log_memory}\n"
        )

        self.bce_dice = LossWrapper(BCEDiceLoss)
        self.boundary = LossWrapper(BoundaryLoss)

        self.best_val_acc = 0.0
        self.best_epoch_id = 0
        self.train_acc = []
        self.val_acc = []

        train_curve_path = os.path.join(self.checkpoint_dir, "train_acc.npy")
        val_curve_path = os.path.join(self.checkpoint_dir, "val_acc.npy")
        if os.path.exists(train_curve_path):
            self.train_acc = np.load(train_curve_path).tolist()
        if os.path.exists(val_curve_path):
            self.val_acc = np.load(val_curve_path).tolist()

    @staticmethod
    def _format_hours(hours: float) -> str:
        if hours < 1:
            return f"{hours * 60:.1f} min"
        return f"{hours:.2f} h"

    @staticmethod
    def _is_dataloader_worker_crash(exc: RuntimeError) -> bool:
        message = str(exc)
        worker_markers = [
            "DataLoader worker",
            "exited unexpectedly",
            "_queue.Empty",
            "worker process",
        ]
        return any(marker in message for marker in worker_markers)

    def _timer_update(self, batch_id: int) -> tuple[float, float]:
        effective_epoch = max(self.epoch_id - self.epoch_to_start, 0)
        current_step = effective_epoch * self.steps_per_epoch + batch_id
        self.global_step = current_step
        progress = min((self.global_step + 1) / max(self.total_steps, 1), 0.999999)
        self.timer.update_progress(progress)
        est = self.timer.estimated_remaining()
        imps = (self.global_step + 1) * self.batch_size / max(self.timer.get_stage_elapsed(), 1e-8)
        return imps, est

    def _load_checkpoint(self, ckpt_name: str = "last_ckpt.pt"):
        ckpt_path = os.path.join(self.checkpoint_dir, ckpt_name)
        if not os.path.exists(ckpt_path):
            self.logger.write("Resume mode: OFF | training from scratch.\n")
            return

        self.logger.write(f"Resume mode: ON | loading last checkpoint from {ckpt_path}...\n")
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.net_G.load_state_dict(checkpoint["model_G_state_dict"])
        self.optimizer_G.load_state_dict(checkpoint["optimizer_G_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if self.use_amp and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        self.epoch_to_start = checkpoint["epoch_id"] + 1
        self.best_val_acc = checkpoint.get("best_val_acc", 0.0)
        self.best_epoch_id = checkpoint.get("best_epoch_id", 0)
        self.total_steps = max(1, (self.args.max_epochs - self.epoch_to_start) * self.steps_per_epoch)

        self.logger.write(
            f"Epoch_to_start = {self.epoch_to_start}, "
            f"Historical_best_acc = {self.best_val_acc:.4f} (at epoch {self.best_epoch_id})\n\n"
        )

    def _progress_message(self) -> str:
        if self.total_steps <= 0:
            return "eta=unknown"
        elapsed_sec = self.timer.get_stage_elapsed()
        if elapsed_sec <= 0 or self.global_step <= 0:
            return "eta=warming_up"
        images_per_sec = self.global_step * self.args.batch_size / max(elapsed_sec, 1e-8)
        if hasattr(self.timer, "estimated_remaining"):
            eta_hours = self.timer.estimated_remaining()
            return f"speed={images_per_sec:.2f} img/s eta={self._format_hours(max(eta_hours, 0.0))}"
        return f"speed={images_per_sec:.2f} img/s eta=unknown"

    def _compute_loss(self, preds, gt):
        loss = 0.0
        gt_4d = gt.unsqueeze(1).float() if gt.dim() == 3 else gt.float()

        if isinstance(preds, (list, tuple)):
            for i, pred in enumerate(preds):
                gt_resized = F.interpolate(gt_4d, size=pred.shape[2:], mode="nearest").squeeze(1).long()
                cur = self.bce_dice(pred, gt_resized)
                if self.args.boundary_weight > 0:
                    cur = cur + self.args.boundary_weight * self.boundary(pred, gt_resized)
                weight = self.args.loss_weights[i] if i < len(self.args.loss_weights) else 1.0
                loss = loss + weight * cur
            loss = loss / len(preds)
        else:
            gt_resized = gt_4d.squeeze(1).long()
            loss = self.bce_dice(preds, gt_resized)
            if self.args.boundary_weight > 0:
                loss = loss + self.args.boundary_weight * self.boundary(preds, gt_resized)
        return loss

    def _metric_update(self, pred_logits, gt):
        pred = torch.argmax(pred_logits.detach(), dim=1)
        return self.metric.update_cm(pr=pred.cpu().numpy(), gt=gt.cpu().numpy())

    def _save_checkpoint(self, epoch_id: int, is_best: bool = False):
        payload = {
            "epoch_id": epoch_id,
            "best_val_acc": self.best_val_acc,
            "best_epoch_id": self.best_epoch_id,
            "model_G_state_dict": self.net_G.state_dict(),
            "optimizer_G_state_dict": self.optimizer_G.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }
        if self.use_amp:
            payload["scaler_state_dict"] = self.scaler.state_dict()
        torch.save(payload, os.path.join(self.checkpoint_dir, "last_ckpt.pt"))
        if is_best:
            torch.save(payload, os.path.join(self.checkpoint_dir, "best_ckpt.pt"))

    def _reset_memory_peak(self):
        if self.device.type == "cuda":
            utils_.reset_cuda_peak_memory_stats(self.device)

    def _log_memory_state(self, tag: str):
        if self.log_memory and self.device.type == "cuda":
            self.logger.write(f"[cuda_memory] {tag}: {utils_.format_cuda_memory_stats(self.device)}\n")

    @staticmethod
    def _should_log_batch(batch_id: int, total_batches: int) -> bool:
        if total_batches <= 0:
            return False
        key_points = {0, total_batches // 2, total_batches - 1}
        return batch_id in key_points

    def _log_batch_state(self, batch_id: int, total_batches: int, loss_value: float, running_mf1: float) -> None:
        imps, est = self._timer_update(batch_id)
        self.logger.write(
            f"Is_training: {self.is_training}. "
            f"[{self.epoch_id},{self.max_num_epochs - 1}]"
            f"[{batch_id + 1},{total_batches}], "
            f"imps: {imps:.2f}, est: {est:.2f}h, "
            f"G_loss: {loss_value:.5f}, running_mf1: {running_mf1:.5f}\n"
        )

    def _log_epoch_state(self, scores: dict[str, float]) -> None:
        epoch_mf1 = scores["mf1"]
        self.logger.write(
            f"Is_training: {self.is_training}. Epoch {self.epoch_id} / {self.max_num_epochs - 1}, "
            f"epoch_mF1= {epoch_mf1:.5f}\n"
        )
        metric_order = [
            "acc",
            "miou",
            "mf1",
            "iou_0",
            "iou_1",
            "F1_0",
            "F1_1",
            "precision_0",
            "precision_1",
            "recall_0",
            "recall_1",
        ]
        metric_line = " ".join([f"{key}: {scores[key]:.5f}" for key in metric_order if key in scores])
        self.logger.write(metric_line + " \n\n")

    def _update_checkpoints(self, epoch_acc: float) -> None:
        historical_best_acc = self.best_val_acc
        historical_best_epoch = self.best_epoch_id
        self._save_checkpoint(self.epoch_id, is_best=False)
        self.logger.write(
            f"Lastest model updated. Epoch_acc={epoch_acc:.4f}, "
            f"Historical_best_acc={historical_best_acc:.4f} (at epoch {historical_best_epoch})\n\n"
        )

        if epoch_acc > self.best_val_acc:
            self.best_val_acc = epoch_acc
            self.best_epoch_id = self.epoch_id
            self._save_checkpoint(self.epoch_id, is_best=True)
            self.logger.write("*" * 10 + "Best model updated!\n\n")

    def _run_epoch(self, epoch_id: int, training: bool):
        self.metric.clear()
        phase = "train" if training else "val"
        loader = self.dataloaders[phase]
        self.net_G.train(training)
        running_loss = 0.0
        total_batches = len(loader)
        self.is_training = training
        self.epoch_id = epoch_id

        try:
            for batch_id, batch in enumerate(loader):
                img1 = batch["A"].to(self.device)
                img2 = batch["B"].to(self.device)
                gt = batch["L"].to(self.device).long()
                if gt.dim() == 4:
                    gt = gt.squeeze(1)

                with torch.set_grad_enabled(training):
                    with utils_.build_autocast_context(self.device, self.use_amp, self.amp_dtype):
                        preds = self.net_G(img1, img2)
                        main_pred = preds[0] if isinstance(preds, (list, tuple)) else preds
                        loss = self._compute_loss(preds, gt)
                    if training:
                        self.optimizer_G.zero_grad(set_to_none=True)
                        if self.use_amp:
                            self.scaler.scale(loss).backward()
                            self.scaler.step(self.optimizer_G)
                            self.scaler.update()
                        else:
                            loss.backward()
                            self.optimizer_G.step()

                running_loss += float(loss.item())
                mf1 = self._metric_update(main_pred, gt)

                if training:
                    utils_.maybe_clear_cuda_cache(
                        step=batch_id,
                        interval=self.cache_clear_interval,
                        gc_collect=False,
                    )

                if self._should_log_batch(batch_id, total_batches):
                    self._log_batch_state(batch_id, total_batches, float(loss.item()), mf1)
        except RuntimeError as exc:
            if training or not self._is_dataloader_worker_crash(exc):
                raise

            self.logger.write(
                f"[dataloader_fallback] Validation loader crashed with num_workers={loader.num_workers}. "
                "Retrying this validation epoch with num_workers=0.\n"
            )
            utils_.maybe_clear_cuda_cache(force=True, gc_collect=True)
            self.dataloaders[phase] = utils_.rebuild_dataloader(
                loader,
                num_workers=0,
                shuffle=False,
                pin_memory=False,
            )
            return self._run_epoch(epoch_id, training=False)

        scores = self.metric.get_scores()
        epoch_loss = running_loss / max(1, len(loader))
        scores["epoch_loss"] = epoch_loss
        return epoch_loss, scores

    def train(self):
        self._load_checkpoint()

        if self.epoch_to_start >= self.args.max_epochs:
            self.logger.write(
                f"checkpoint already reached max_epochs={self.args.max_epochs}, skip training.\n\n"
            )
            return

        remaining_epochs = self.args.max_epochs - self.epoch_to_start
        self.logger.write(
            f"Training plan | device={self.device} | steps_per_epoch={self.steps_per_epoch} | "
            f"remaining_epochs={remaining_epochs} | total_steps_remaining={self.total_steps}\n"
        )
        self.logger.write(f"Checkpoint dir: {self.checkpoint_dir}\nVis dir: {self.vis_dir}\n\n")

        for epoch_id in range(self.epoch_to_start, self.args.max_epochs):
            epoch_started = time.time()
            utils_.maybe_clear_cuda_cache(force=True, gc_collect=True)
            self._reset_memory_peak()
            self._log_memory_state(f"epoch_{epoch_id}_train_start")
            self.logger.write(f"lr={self.optimizer_G.param_groups[0]['lr']:.7f}\n")
            _, train_scores = self._run_epoch(epoch_id, training=True)
            self._log_epoch_state(train_scores)
            self.scheduler.step()
            self._log_memory_state(f"epoch_{epoch_id}_train_end")

            utils_.maybe_clear_cuda_cache(force=True, gc_collect=True)

            self.logger.write("Begin evaluation...\n")
            self._reset_memory_peak()
            self._log_memory_state(f"epoch_{epoch_id}_val_start")
            _, val_scores = self._run_epoch(epoch_id, training=False)
            self._log_epoch_state(val_scores)
            self._log_memory_state(f"epoch_{epoch_id}_val_end")

            utils_.maybe_clear_cuda_cache(force=True, gc_collect=True)

            train_mf1 = train_scores["mf1"]
            val_mf1 = val_scores["mf1"]
            self.train_acc.append(train_mf1)
            self.val_acc.append(val_mf1)
            np.save(os.path.join(self.checkpoint_dir, "train_acc.npy"), np.array(self.train_acc, np.float32))
            np.save(os.path.join(self.checkpoint_dir, "val_acc.npy"), np.array(self.val_acc, np.float32))

            self._update_checkpoints(val_mf1)

            epoch_minutes = (time.time() - epoch_started) / 60.0
            remaining_after_this = self.args.max_epochs - epoch_id - 1
            rough_remaining_hours = epoch_minutes * remaining_after_this / 60.0
            self.logger.write(
                utils_.format_epoch_summary(
                    epoch_id=epoch_id,
                    max_epochs=self.args.max_epochs,
                    train_mf1=train_mf1,
                    val_mf1=val_mf1,
                    epoch_minutes=epoch_minutes,
                    remaining_hours=rough_remaining_hours,
                    best_epoch=self.best_epoch_id,
                    best_val_mf1=self.best_val_acc,
                    amp_enabled=self.use_amp,
                    amp_dtype=self.amp_dtype,
                    device=self.device,
                ) + "\n\n"
            )


class AblationEvaluator:
    def __init__(self, args, dataloader):
        self.args = args
        self.dataloader = dataloader
        self.device = torch.device(
            f"cuda:{args.gpu_ids[0]}" if torch.cuda.is_available() and len(args.gpu_ids) > 0 else "cpu"
        )
        self.net_G = build_ablation_model(args).to(self.device)
        self.metric = ConfuseMatrixMeter(n_class=args.n_class)
        self.log_path = os.path.join(args.checkpoint_dir, "log_eval.txt")
        ensure_log_writable(self.log_path)
        self.logger = Logger(self.log_path)
        self.logger.write_dict_str(args.__dict__)
        self.batch = None
        self.batch_id = 0
        self.vis_count = 0
        self.global_fp_map = None
        self.global_fn_map = None

    def load_checkpoint(self, checkpoint_name="best_ckpt.pt"):
        ckpt_path = os.path.join(self.args.checkpoint_dir, checkpoint_name)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Checkpoint not found for evaluation: {ckpt_path}. "
                "Please finish training first or reuse the same project_name that already has a checkpoint."
            )
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.net_G.load_state_dict(checkpoint["model_G_state_dict"])
        best_val_acc = checkpoint.get("best_val_acc", 0.0)
        best_epoch_id = checkpoint.get("best_epoch_id", 0)
        self.logger.write(f"Eval Historical_best_acc = {best_val_acc:.4f} (at epoch {best_epoch_id})\n\n")

    def _accumulate_error(self, pred: torch.Tensor, gt: torch.Tensor) -> None:
        fp = ((pred == 1) & (gt == 0)).float()
        fn = ((pred == 0) & (gt == 1)).float()

        if self.global_fp_map is None:
            _, h, w = fp.shape
            self.global_fp_map = torch.zeros((h, w), device=self.device)
            self.global_fn_map = torch.zeros((h, w), device=self.device)

        self.global_fp_map += fp.sum(dim=0)
        self.global_fn_map += fn.sum(dim=0)

    def _collect_running_batch_states(self, pred: torch.Tensor, gt: torch.Tensor, running_acc: float) -> None:
        vis_interval = 20
        max_vis_num = 50
        if (self.batch_id % vis_interval != 0) or (self.vis_count >= max_vis_num):
            return

        vis_input = utils_.make_numpy_grid(de_norm(self.batch["A"]))
        vis_input2 = utils_.make_numpy_grid(de_norm(self.batch["B"]))
        vis_pred = utils_.make_numpy_grid(pred.unsqueeze(1).float().cpu() * 255)
        vis_gt = utils_.make_numpy_grid(gt.unsqueeze(1).float().cpu() * 255 if gt.max() <= 1 else gt.unsqueeze(1).cpu())

        fp = ((pred == 1) & (gt == 0)).float()
        fn = ((pred == 0) & (gt == 1)).float()
        vis_fp = utils_.make_numpy_grid(fp.unsqueeze(1).cpu() * 255)
        vis_fn = utils_.make_numpy_grid(fn.unsqueeze(1).cpu() * 255)

        vis = np.concatenate([vis_input, vis_input2, vis_pred, vis_gt, vis_fp, vis_fn], axis=0)
        vis = np.clip(vis, 0.0, 1.0)
        file_name = os.path.join(self.args.vis_dir, f"eval_batch_{self.batch_id}_F1_{running_acc:.4f}.jpg")
        plt.imsave(file_name, vis)
        self.vis_count += 1

    def _generate_heatmap(self) -> None:
        if self.global_fp_map is None or self.global_fn_map is None:
            self.logger.write("No accumulated FP/FN map found; skip heatmap generation.\n")
            return

        self.logger.write(">>> Generating global error heatmap & Distribution Analysis...\n")
        fp_np = self.global_fp_map.cpu().numpy()
        fn_np = self.global_fn_map.cpu().numpy()
        total_images = len(self.dataloader.dataset)
        self.logger.write(f"Total evaluated images: {total_images}\n")
        self.logger.write(
            f"[FP Distribution] Max accum: {fp_np.max()}, Mean: {fp_np.mean():.4f}, Std: {fp_np.std():.4f}\n"
        )
        self.logger.write(
            f"[FN Distribution] Max accum: {fn_np.max()}, Mean: {fn_np.mean():.4f}, Std: {fn_np.std():.4f}\n"
        )

        fp_norm = fp_np / (fp_np.max() + 1e-8)
        fn_norm = fn_np / (fn_np.max() + 1e-8)

        plt.figure(figsize=(8, 6))
        plt.imshow(fp_norm, cmap="jet")
        plt.colorbar(label="Normalized Accumulation")
        plt.title(f"False Positive (FP) Heatmap (Max: {fp_np.max():.0f})")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(self.args.vis_dir, "global_fp_heatmap.png"), dpi=300)
        plt.close()

        plt.figure(figsize=(8, 6))
        plt.imshow(fn_norm, cmap="jet")
        plt.colorbar(label="Normalized Accumulation")
        plt.title(f"False Negative (FN) Heatmap (Max: {fn_np.max():.0f})")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(self.args.vis_dir, "global_fn_heatmap.png"), dpi=300)
        plt.close()

    def evaluate(self):
        self.load_checkpoint()
        self.net_G.eval()
        self.logger.write("Begin evaluation...\n")

        while True:
            self.metric.clear()
            self.global_fp_map = None
            self.global_fn_map = None
            self.vis_count = 0
            total_batches = len(self.dataloader)
            print(f"Evaluation plan | total_batches={total_batches} | checkpoint_dir={self.args.checkpoint_dir}")

            try:
                with torch.no_grad():
                    for batch_id, batch in enumerate(self.dataloader):
                        self.batch = batch
                        self.batch_id = batch_id
                        img1 = batch["A"].to(self.device)
                        img2 = batch["B"].to(self.device)
                        gt = batch["L"].to(self.device).long()
                        if gt.dim() == 4:
                            gt = gt.squeeze(1)
                        pred_logits = self.net_G(img1, img2)
                        if isinstance(pred_logits, (list, tuple)):
                            pred_logits = pred_logits[0]
                        pred = torch.argmax(pred_logits, dim=1)
                        running_acc = self.metric.update_cm(pr=pred.cpu().numpy(), gt=gt.cpu().numpy())
                        self._accumulate_error(pred, gt)
                        self._collect_running_batch_states(pred, gt, running_acc)
                        if AblationTrainer._should_log_batch(batch_id, total_batches):
                            print(f"[eval] batch={batch_id + 1}/{total_batches} running_mF1={running_acc:.5f}")
                break
            except RuntimeError as exc:
                if self.dataloader.num_workers == 0 or not AblationTrainer._is_dataloader_worker_crash(exc):
                    raise
                self.logger.write(
                    f"[dataloader_fallback] Test loader crashed with num_workers={self.dataloader.num_workers}. "
                    "Retrying evaluation once with num_workers=0.\n"
                )
                utils_.maybe_clear_cuda_cache(force=True, gc_collect=True)
                self.dataloader = utils_.rebuild_dataloader(
                    self.dataloader,
                    num_workers=0,
                    shuffle=False,
                    pin_memory=False,
                )

        scores = self.metric.get_scores()
        message = "==================== Final Eval Results ====================\n"
        for key, value in scores.items():
            message += f"{key}: {value:.5f}\n"
        self.logger.write(message + "\n")
        self._generate_heatmap()
        self.logger.write("Evaluation Finished.\n")
        return scores


def train_and_eval(args):
    lock_path = acquire_experiment_lock(args.checkpoint_dir)
    try:
        dataloaders = utils_.get_loaders(args)
        trainer = AblationTrainer(args=args, dataloaders=dataloaders)
        trainer.train()

        test_loader = utils_.get_loader(
            args.data_name,
            img_size=args.img_size,
            batch_size=args.batch_size,
            is_train=False,
            split="test",
            num_workers=args.num_workers,
            data_root=getattr(args, "data_root", None),
        )
        evaluator = AblationEvaluator(args=args, dataloader=test_loader)
        scores = evaluator.evaluate()
        print(f"[Ablation:{args.ablation_case}] {scores}")
        return scores
    finally:
        release_experiment_lock(lock_path)
