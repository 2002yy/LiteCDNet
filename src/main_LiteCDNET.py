from argparse import ArgumentParser
import torch
import os

from models.trainer import *
from utils_ import str2bool
import utils_

print(torch.cuda.is_available())

"""
Main function for training LiteCDNet
"""

def train(args):
    dataloaders = utils_.get_loaders(args)
    model = CDTrainer(args=args, dataloaders=dataloaders)
    model.train_models(args=args)


def test(args):
    from models.evaluator import CDEvaluator

    dataloader = utils_.get_loader(
        args.data_name,
        img_size=args.img_size,
        batch_size=args.batch_size,
        is_train=False,
        split='test',
        data_root=args.data_root,
    )

    model = CDEvaluator(args=args, dataloader=dataloader)
    model.eval_models()


if __name__ == '__main__':
    parser = ArgumentParser()

    # ========= GPU =========
    parser.add_argument('--gpu_ids', type=str, default='0')

    # ========= Project =========
    parser.add_argument(
        '--project_name',
        default='LEVIR_LiteCDNet_BCEDiceBoundary0.3_AdamW_Cosine_150',
        type=str
    )
    parser.add_argument('--checkpoint_root', default='checkpoints', type=str)

    # ========= Data =========
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--dataset', default='CDDataset', type=str)
    parser.add_argument('--data_name', default='LEVIR', type=str)
    parser.add_argument('--data_root', default=None, type=str,
                        help='Optional dataset root override. Defaults to repo-local data/<dataset>.')

    parser.add_argument('--batch_size', default=3, type=int)   # ⭐建议提高
    parser.add_argument('--split', default="train", type=str)
    parser.add_argument('--split_val', default="val", type=str)
    parser.add_argument('--img_size', default=256, type=int)

    # ========= Model =========
    parser.add_argument('--n_class', default=2, type=int)
    parser.add_argument('--embed_dim', default=64, type=int)

    # 🔥 关键：改成你的模型名字（必须和 define_G 对应）
    parser.add_argument('--net_G', default='LiteCDNet', type=str)

    parser.add_argument('--backbone', default='mobilenetv2', type=str)
    parser.add_argument('--mode', default='None', type=str)

    # ✅ LiteCDNet 必开
    parser.add_argument('--deep_supervision', default=True, type=str2bool)

    # ❌ 不用 IFNet 模式
    parser.add_argument('--loss_SD', default=False, type=str2bool)
    # ========= Loss =========
    parser.add_argument('--boundary_weight', default=0.3, type=float)

    # 多尺度监督权重，按 mask1, mask2, mask3, mask4 顺序传
    parser.add_argument(
        '--loss_weights',
        nargs='+',
        type=float,
        default=[1.0, 0.7, 0.5, 0.3]
    )

    # ========= Optimizer =========
    parser.add_argument('--optimizer', default='adamw', type=str)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--max_epochs', default=150, type=int)
    parser.add_argument('--use_amp', default=False, type=str2bool,
                        help='Enable mixed precision training on CUDA.')
    parser.add_argument('--amp_dtype', default='fp16', type=str, choices=['fp16', 'bf16'],
                        help='AMP compute dtype.')
    parser.add_argument('--cache_clear_interval', default=10, type=int,
                        help='Clear CUDA cache every N train batches. Set 0 to disable.')
    parser.add_argument('--log_memory', default=True, type=str2bool,
                        help='Log CUDA allocated/reserved memory at epoch boundaries.')

    # ❌ 已不用（保留不影响）
    parser.add_argument('--lr_policy', default='cosine', type=str)

    args = parser.parse_args()

    # ========= GPU处理（关键修复）=========
    args.gpu_ids = [int(x) for x in args.gpu_ids.split(',') if int(x) >= 0]
    utils_.get_device(args)

    print("Using GPUs:", args.gpu_ids)

    # ========= paths =========
    args.checkpoint_dir = os.path.join(args.checkpoint_root, args.project_name)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    args.vis_dir = os.path.join('vis', args.project_name)
    os.makedirs(args.vis_dir, exist_ok=True)

    # ========= train =========
    train(args)

    # ========= test =========
    test(args)
