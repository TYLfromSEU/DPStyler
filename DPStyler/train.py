import argparse
import copy
import torch
import os
from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer

from yacs.config import CfgNode as CN


def print_args(args, cfg):
    print('***************')
    print('** Arguments **')
    print('***************')
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print('{}: {}'.format(key, args.__dict__[key]))
    print('************')
    print('** Config **')
    print('************')
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head

    if args.arcface_s:
        cfg.ARCFACE_S=args.arcface_s

    if args.arcface_m:
        cfg.ARCFACE_M=args.arcface_m

    if args.num_styles:
        cfg.TRAINER.NUM_STYLES =args.num_styles

    if args.txts_path:
        cfg.TXTS_PATH=args.txts_path

    if args.eval_epoch:
        cfg.TRAINER.EVAL_EPOCH=args.eval_epoch

    if args.refresh:
        cfg.TRAINER.REFRESH=args.refresh


def extend_cfg(cfg):
    # Here you can extend the existing cfg variables by adding new ones
    cfg.SOMENAME = CN()


def setup_cfg(args):
    cfg = get_cfg_default()
    add_stn_config(cfg)
    extend_cfg(cfg)
    reset_cfg(cfg, args)
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    #add_stn_config(cfg)
    cfg.freeze()
    return cfg

def add_stn_config(cfg):
    cfg.MODEL.HEAD.REDUCTION_RATIO=2

def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print('Setting fixed seed: {}'.format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print('Collecting env info ...')
    print('** System info **\n{}\n'.format(collect_env_info()))

    trainer = build_trainer(cfg)

    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
        return

    if not args.no_train:
        trainer.train()


    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='', help='path to dataset')
    parser.add_argument(
        '--output-dir', type=str, default='', help='output directory'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default='',
        help='checkpoint directory (from which the training resumes)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=-1,
        help='only positive value enables a fixed seed'
    )
    parser.add_argument(
        '--source-domains',
        type=str,
        nargs='+',
        help='source domains for DA/DG'
    )
    parser.add_argument(
        '--target-domains',
        type=str,
        nargs='+',
        help='target domains for DA/DG'
    )
    parser.add_argument(
        '--use_cuda',
        type=bool,
        nargs='+',
        help='usr_cuda or not'
    )
    parser.add_argument(
        '--gpu',
        type=int,
        nargs='+',
        help='idx of gpu'
    )
    parser.add_argument(
        '--transforms', type=str, nargs='+', help='data augmentation methods'
    )
    parser.add_argument(
        '--config-file', type=str, default='', help='path to config file'
    )
    parser.add_argument(
        '--dataset-config-file',
        type=str,
        default='',
        help='path to config file for dataset setup'
    )
    parser.add_argument(
        '--trainer', type=str, default='', help='name of trainer'
    )
    parser.add_argument(
        '--backbone', type=str, default='', help='name of CNN backbone'
    )
    parser.add_argument('--head', type=str, default='', help='name of head')
    parser.add_argument(
        '--eval-only', action='store_true', help='evaluation only'
    )
    parser.add_argument(
        '--model-dir',
        type=str,
        default='',
        help='load model from this directory for eval-only mode'
    )
    parser.add_argument(
        '--load-epoch',
        type=int,
        help='load model weights at this epoch for evaluation'
    )
    parser.add_argument(
        '--no-train', action='store_true', help='do not call trainer.train()'
    )
    parser.add_argument(
        'opts',
        default=None,
        nargs=argparse.REMAINDER,
        help='modify config options using the command-line'
    )
    parser.add_argument(
        '--arcface_s',
        type=float,
        default=5,
        help='arcface s'
    )
    parser.add_argument(
        '--arcface_m',
        type=float,
        default=0.5,
        help='arcface m'
    )
    parser.add_argument(
        '--num_styles',
        type=int,
        default=80,
        help='random styles'
    )
    parser.add_argument(
        '--txts_path',
        type=str,
        default='',
        help='txt path'
    )
    parser.add_argument(
        '--eval_epoch',
        type=int,
        default=0,
        help='eval_after'
    )
    parser.add_argument(
        '--refresh',
        type=str,
        default='',
        help='refresh style every epoch'
    )

    args = parser.parse_args()
    main(args)
