from argparse import ArgumentParser
from pathlib import Path
from typing import Optional, List

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import lpips

from datasets.fran_dataset import FRANDataset
from models.fran import FRAN
from models.patchGAN import PatchGAN
from transforms.fran_transforms import data_transforms
from utils.epochs import training_epoch, validation_epoch
from utils.running_extrema import RunningExtrema, MAX, MIN


def run_training(
    lr: float,
    beta1: float,
    beta2: float,
    weight_decay: float,

    device: torch.device,

    best_metric: str,
    higher_is_better: bool,
    ckpts_path: Path,
    run_name: str,

    val_fold: int,
    num_folds: int,

    crop_size: int,
    norm_mean: List[float],
    norm_std: List[float],
    jitter_brightness: float,
    jitter_saturation: float,
    jitter_hue: float,
    jitter_contrast: float,
    random_angle: float,

    l1_weight: float,
    lpips_weight: float,
    adv_weight: float,

    discr_steps: int,

    num_epochs: int,
    val_every: int,
    num_val: int,
    batch_size: int,
    val_batch_size: int,
    num_workers: int,

    load_ckpt: Optional[Path],
    save_last: bool,
    save_best: bool,

    data_root: str,
):
    # Create datasets
    tfm_train, tfm_val = data_transforms(
        crop_size=crop_size,
        norm_mean=norm_mean,
        norm_std=norm_std,
        jitter_brightness=jitter_brightness,
        jitter_saturation=jitter_saturation,
        jitter_hue=jitter_hue,
        jitter_contrast=jitter_contrast,
        random_angle=random_angle,
    )
    ds_train = FRANDataset(
        data_root=data_root,
        is_val=False,
        transform=tfm_train,
        num_folds=num_folds,
        val_fold=val_fold,
    )
    ds_val = FRANDataset(
        data_root=data_root,
        is_val=True,
        transform=tfm_val,
        num_folds=num_folds,
        val_fold=val_fold,
    )

    # Create data loaders
    dl_train = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    # Create models
    fran = FRAN()
    discr = PatchGAN(in_channels=4)  # RGB + age channel

    # Load checkpoint
    if load_ckpt is not None:
        state_dicts = torch.load(load_ckpt)
        fran.load_state_dict(state_dicts['FRAN'])
        discr.load_state_dict(state_dicts['PatchGAN'])

    # Move models to device
    fran = fran.to(device)
    discr = discr.to(device)

    # Define optimizer
    fran_optimizer = Adam(
        fran.parameters(),
        lr=lr,
        betas=(beta1, beta2),
        weight_decay=weight_decay
    )
    discr_optimizer = Adam(
        discr.parameters(),
        lr=lr,
        betas=(beta1, beta2),
        weight_decay=weight_decay
    )

    # Loss functions
    l1_loss_fn = nn.L1Loss()
    lpips_loss_fn = lpips.LPIPS(net='vgg').to(device)
    bce_loss_fn = nn.BCELoss()

    # Initialize Running Extrema
    running_extrema_best = RunningExtrema(
        MAX if higher_is_better else MIN
    )
    running_extrema_worst = RunningExtrema(
        MIN if higher_is_better else MAX
    )
    ckpt_dir = Path(ckpts_path)

    running_extrema_best.clear()
    running_extrema_worst.clear()

    # Training loop
    for epoch_idx in tqdm(range(num_epochs), leave=True):
        is_end_of_epoch = False

        while not is_end_of_epoch:
            # Training steps
            fran.train()
            discr.train()
            is_end_of_epoch = training_epoch(
                fran=fran,
                discr=discr,

                fran_optim=fran_optimizer,
                discr_optim=discr_optimizer,

                l1_loss_fn=l1_loss_fn,
                lpips_loss_fn=lpips_loss_fn,
                bce_loss_fn=bce_loss_fn,

                l1_weight=l1_weight,
                lpips_weight=lpips_weight,
                adv_weight=adv_weight,

                epoch_idx=epoch_idx,
                device=device,
                dl_train=dl_train,
                discr_steps=discr_steps,

                num_steps=val_every
            )

            # Validation epoch
            fran.eval()
            discr.eval()
            validation_epoch(
                fran=fran,
                discr=discr,

                l1_loss_fn=l1_loss_fn,
                lpips_loss_fn=lpips_loss_fn,
                bce_loss_fn=bce_loss_fn,

                l1_weight=l1_weight,
                lpips_weight=lpips_weight,
                adv_weight=adv_weight,

                epoch_idx=epoch_idx,
                device=device,
                dl_val=dl_val,

                running_extrema_best=running_extrema_best,
                running_extrema_worst=running_extrema_worst,

                save_last=save_last,
                save_best=save_best,
                best_metric=best_metric,
                ckpt_dir=ckpt_dir,
                run_name=run_name,

                num_steps=num_val,
            )


def float_list_arg_type(arg):
    return [float(s.strip()) for s in arg.split(',') if len(s.strip()) > 0]


if __name__ == '__main__':
    parser = ArgumentParser()

    # Ckpt
    parser.add_argument(
        '--load_ckpt', default=None,
        help='The path to load model checkpoint weights from.'
    )
    parser.add_argument(
        '--save_best', action='store_true',
        help='If set, save a checkpoint containg the weights with the '
        'best performance, as defined by --best_metric and --higher_is_better.'
    )
    parser.add_argument(
        '--save_last', action='store_true',
        help='If set, save a checkpoint containing the weights of the '
        'last epoch.'
    )
    parser.add_argument(
        '--best_metric', default='Loss_Total',
        help='If this metric improves, create a checkpoint '
        '(when --save_best is set).'
    )
    parser.add_argument(
        '--higher_is_better', action='store_true',
        help='If set, the metric set with --best_metric is better when '
        'it increases.'
    )
    parser.add_argument(
        '--ckpts_path', default='./ckpts',
        help='The directory to save checkpoints.'
    )

    # K-Fold args
    parser.add_argument(
        '--num_folds', default=5,
        help='The number of folds to use.',
        type=int
    )
    parser.add_argument(
        '--val_fold', default=0,
        help='The index of the validation fold.',
        type=int
    )

    # Data root
    parser.add_argument(
        '--data_root',
        default='data/FRAN_dataset',
        help='Directory of the dataset'
    )

    # Dataloader args
    parser.add_argument(
        '--batch_size',
        default=8,
        help='The training batch size.',
        type=int
    )
    parser.add_argument('--val_batch_size', default=8,
                        help='The validation batch size.', type=int)
    parser.add_argument(
        '--num_workers', default=8,
        help='The number of workers to use for data loading.',
        type=int
    )

    # Optimizer args
    parser.add_argument('--lr', default=0.01,
                        help='The learning rate.',
                        type=float)
    parser.add_argument('--beta1', default=0.9,
                        help='Coefficient used for computing running average '
                        'of gradient in Adam.',
                        type=float)
    parser.add_argument('--beta2', default=0.999,
                        help='Coefficient used for computing running average '
                        'of square gradient in Adam.',
                        type=float)
    parser.add_argument('--weight_decay', default=1e-5,
                        help='The weight decay.',
                        type=float)

    # Train args
    parser.add_argument(
        '--num_epochs', default=50,
        help='The number of epochs to train.',
        type=int
    )
    parser.add_argument(
        '--val_every', default=1000,
        help='Run validation epoch after this number of training steps.',
        type=int
    )
    parser.add_argument(
        '--num_val', default=100,
        help='Run this number of validation steps in each validation epoch',
        type=int
    )
    parser.add_argument(
        '--discr_steps', default=5,
        help='The number of discriminator training steps before a FRAN '
        'trainig step.',
        type=int
    )

    # Log args
    parser.add_argument(
        '--wandb_entity', help='Weights and Biases entity.'
    )
    parser.add_argument(
        '--wandb_project', help='Weights and Biases project.'
    )

    # Device arg
    parser.add_argument(
        '--device',
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='The device (cuda/cpu) to use.'
    )

    # Data augmentation
    parser.add_argument(
        '--crop_size',
        default=512,
        help='Crop size to use in the data transform pipeline.',
        type=int,
    )
    parser.add_argument(
        '--jitter_brightness',
        default=0,
        type=int,
        help='Brightness jitter'
    )
    parser.add_argument(
        '--jitter_saturation',
        default=0,
        type=int,
        help='Saturation jitter'
    )
    parser.add_argument(
        '--jitter_hue',
        default=0,
        type=int,
        help='Hue jitter'
    )
    parser.add_argument(
        '--jitter_contrast',
        default=0,
        type=int,
        help='Contrast jitter'
    )
    parser.add_argument(
        '--random_angle',
        default=0,
        type=float,
        help='Select random rotation from range (-random_angle, +random_angle)'
    )
    parser.add_argument(
        '--norm_mean',
        default=[0.5, 0.5, 0.5],
        help='The mean to subtract during data normalization.',
        type=float_list_arg_type,
    )
    parser.add_argument(
        '--norm_std',
        default=[0.5, 0.5, 0.5],
        help='The standard deviation to divide by during data normalization.',
        type=float_list_arg_type,
    )

    # Loss term weights
    parser.add_argument(
        '--l1_weight',
        default=1.0,
        type=float,
        help='Weight for the L1 loss term.'
    )
    parser.add_argument(
        '--lpips_weight',
        default=1.0,
        type=float,
        help='Weight for the LPIPS loss term.'
    )
    parser.add_argument(
        '--adv_weight',
        default=0.05,
        type=float,
        help='Weight for the adversarial loss term.'
    )

    args = parser.parse_args()

    args_dict = vars(args)
    wandb.init(entity=args.wandb_entity, project=args.wandb_project,
               config=args_dict)

    del args_dict['wandb_entity']
    del args_dict['wandb_project']

    run_training(
        **vars(args),
        run_name=wandb.run.id,
    )
