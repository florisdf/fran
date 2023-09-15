from pathlib import Path
import pandas as pd
import sys
from typing import Callable, Optional

from PIL import Image
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
import wandb

from .log import log
from .ckpts import create_checkpoints
from .running_extrema import RunningExtrema

train_batch_idx = -1  # should have global scope


def training_epoch(
    fran,
    discr,

    fran_optim: Optimizer,
    discr_optim: Optimizer,

    l1_loss_fn,
    lpips_loss_fn,
    bce_loss_fn,

    l1_weight: float,
    lpips_weight: float,
    adv_weight: float,

    epoch_idx: int,
    device: torch.device,
    dl_train: DataLoader,
    discr_steps: int,

    num_steps: int = None,
):
    global train_batch_idx
    num_steps = num_steps if num_steps is not None else len(dl_train)

    start_idx = train_batch_idx + 1
    end_idx = start_idx + min(num_steps, len(dl_train) - start_idx % len(dl_train))

    for (train_batch_idx, batch) in zip(
        tqdm(range(start_idx, end_idx), leave=False),
        dl_train
    ):
        src_img, src_age, tgt_img, tgt_age = batch
        src_img = src_img.to(device)
        src_age = src_age.to(device)

        tgt_img = tgt_img.to(device)
        tgt_age = tgt_age.to(device)

        # Predict image with tgt age
        fran_img = fran(src_img, src_age, tgt_age)

        log_dict = {}

        if train_batch_idx % discr_steps == 0:
            # Train FRAN

            # L1 Loss
            l1_loss = l1_loss_fn(fran_img, tgt_img)

            # LPIPS loss
            lpips_loss = lpips_loss_fn(fran_img, tgt_img).mean()

            # Adversarial loss (note: FRAN wants discr's outputs to be 1)
            pred_score = torch.sigmoid(discr(fran_img, tgt_age))
            adv_loss = bce_loss_fn(pred_score, torch.ones_like(pred_score))

            # Sum losses
            fran_loss = (
                l1_weight * l1_loss
                + lpips_weight * lpips_loss
                + adv_weight * adv_loss
            )

            # Take optimization step
            fran.zero_grad()
            fran_loss.backward()
            fran_optim.step()

            log_dict.update({
                'FRAN_L1': l1_loss,
                'FRAN_LPIPS': lpips_loss,
                'FRAN_Adv': adv_loss,
                'FRAN_Total': fran_loss,
            })

            if torch.isnan(fran_loss):
                sys.exit('Loss is NaN. Exiting...')

        # Train discriminator
        # Loss for fake images
        pred_score_fake = torch.sigmoid(discr(fran_img.detach(), tgt_age))
        loss_fake = bce_loss_fn(pred_score_fake, torch.zeros_like(pred_score_fake))

        # Losses for real images with true age
        pred_score_real1 = torch.sigmoid(discr(src_img, src_age))
        loss_real1 = bce_loss_fn(pred_score_real1, torch.ones_like(pred_score_real1))

        pred_score_real2 = torch.sigmoid(discr(tgt_img, tgt_age))
        loss_real2 = bce_loss_fn(pred_score_real2, torch.ones_like(pred_score_real2))

        # Also compute loss for real images with wrong ages; these should also
        # give a score of zero
        wrong_age = get_wrong_ages(src_age, dl_train.dataset.max_age,
                                   dl_train.dataset.min_age)
        pred_score_wrong = torch.sigmoid(discr(src_img, wrong_age))
        loss_wrong = bce_loss_fn(pred_score_wrong, torch.zeros_like(pred_score_wrong))

        # Sum losses
        discr_loss = loss_fake + loss_real1 + loss_real2 + loss_wrong

        # Take optimization step
        discr.zero_grad()
        discr_loss.backward()
        discr_optim.step()

        # Log results
        log_dict.update({
            'Discr_fake': loss_fake,
            'Discr_real1': loss_real1,
            'Discr_real2': loss_real2,
            'Discr_wrong_age': loss_wrong,
            'Discr_total': discr_loss,
        })
        log(log_dict, epoch_idx=epoch_idx,
            batch_idx=train_batch_idx,
            section='Train')

        if torch.isnan(discr_loss):
            sys.exit('Loss is NaN. Exiting...')

    # Return True when an epoch has ended
    return end_idx % len(dl_train) == 0


def get_wrong_ages(true_ages, max_age, min_age):
    age_grid = torch.stack([torch.randperm(max_age - min_age + 1) + min_age
                          for _ in range(len(true_ages))]).type_as(true_ages)
    age_grid = age_grid[age_grid != true_ages[:, None]].reshape((age_grid.shape[0], age_grid.shape[1] - 1))
    return age_grid[:, 0]


@torch.no_grad()
def validation_epoch(
    fran,
    discr,

    l1_loss_fn,
    lpips_loss_fn,
    bce_loss_fn,

    l1_weight: float,
    lpips_weight: float,
    adv_weight: float,

    epoch_idx: int,
    device: torch.device,
    dl_val: DataLoader,

    running_extrema_best: RunningExtrema,
    running_extrema_worst: RunningExtrema,

    save_last: bool,
    save_best: bool,
    best_metric: str,
    ckpt_dir: Path,
    run_name: str,

    num_steps: int = None,
    log_img_every: int = 5,
    img_log_size: tuple = (200, 100),
):
    log_dicts = []
    wandb_ims = []

    num_steps = num_steps if num_steps is not None else len(dl_val)
    num_steps = min(num_steps, len(dl_val))

    for (val_batch_idx, batch) in zip(
        tqdm(range(num_steps), leave=False),
        dl_val
    ):
        src_img, src_age, tgt_img, tgt_age = batch
        src_img = src_img.to(device)
        src_age = src_age.to(device)

        tgt_img = tgt_img.to(device)
        tgt_age = tgt_age.to(device)

        # Predict image with tgt age
        fran_img = fran(src_img, src_age, tgt_age)

        log_dict = {}

        # L1 Loss
        l1_loss = l1_loss_fn(fran_img, tgt_img)

        # LPIPS loss
        lpips_loss = lpips_loss_fn(fran_img, tgt_img).mean()

        # Adversarial loss (note: FRAN wants discr's outputs to be 1)
        pred_score = torch.sigmoid(discr(fran_img, tgt_age))
        adv_loss = bce_loss_fn(pred_score, torch.ones_like(pred_score))

        # Sum losses
        fran_loss = (
            l1_weight * l1_loss
            + lpips_weight * lpips_loss
            + adv_weight * adv_loss
        )

        log_dict.update({
            'FRAN_L1': l1_loss,
            'FRAN_LPIPS': lpips_loss,
            'FRAN_Adv': adv_loss,
            'FRAN_Total': fran_loss,
        })

        # Loss for fake images
        pred_score_fake = torch.sigmoid(discr(fran_img.detach(), tgt_age))
        loss_fake = bce_loss_fn(pred_score_fake, torch.zeros_like(pred_score_fake))

        # Losses for real images with true age
        pred_score_real1 = torch.sigmoid(discr(src_img, src_age))
        loss_real1 = bce_loss_fn(pred_score_real1, torch.ones_like(pred_score_real1))

        pred_score_real2 = torch.sigmoid(discr(tgt_img, tgt_age))
        loss_real2 = bce_loss_fn(pred_score_real2, torch.ones_like(pred_score_real2))

        # Also compute loss for real images with wrong ages; these should also
        # give a score of zero
        wrong_age = get_wrong_ages(src_age, dl_val.dataset.max_age,
                                   dl_val.dataset.min_age)
        pred_score_wrong = torch.sigmoid(discr(src_img, wrong_age))
        loss_wrong = bce_loss_fn(pred_score_wrong, torch.zeros_like(pred_score_wrong))

        # Sum losses
        discr_loss = loss_fake + loss_real1 + loss_real2 + loss_wrong

        # Log results
        log_dict.update({
            'Discr_fake': loss_fake,
            'Discr_real1': loss_real1,
            'Discr_real2': loss_real2,
            'Discr_wrong_age': loss_wrong,
            'Discr_Total': discr_loss,
        })
 
        log_dict.update({
            'Loss_Total': fran_loss + discr_loss,
        })

        log_dicts.append(log_dict)

        if val_batch_idx % log_img_every == 0:
            # Log last validation batch
            norm = dl_val.dataset.transform.transforms[-1]
            inv_std = [1/s for s in norm.std]
            inv_mean = [-m/s for m, s in zip(norm.mean, norm.std)]
            inv_norm = Normalize(inv_mean, inv_std)

            img1, img2, age1, age2 = list(zip(
                src_img, fran_img.detach(),
                src_age, tgt_age,
            ))[0]

            wandb_ims.append(
                wandb.Image(to_pil_image(torch.cat([inv_norm(img1), inv_norm(img2)], dim=-1).cpu()).resize(img_log_size),
                            caption=f'{age1} -> {age2}')
            )

    wandb.log({
        'Examples': wandb_ims,
        'batch_idx': train_batch_idx,
    })

    val_log_dict = pd.DataFrame(log_dicts).mean().to_dict()
    log(val_log_dict, epoch_idx=epoch_idx,
        batch_idx=train_batch_idx,
        section='Val')

    # Check if the value of the metric to optimize is the best
    best_metric_val = val_log_dict[best_metric]
    is_best = running_extrema_best.is_new_extremum(best_metric,
                                                   best_metric_val)
    # Create checkpoints
    create_checkpoints(fran, discr, run_name, ckpt_dir,
                       save_best=save_best and is_best,
                       save_last=save_last)

    # Update and log running extrema
    running_extrema_best.update_dict(val_log_dict)
    running_extrema_worst.update_dict(val_log_dict)

    log(running_extrema_best.extrema_dict, epoch_idx=epoch_idx,
        section=f'{running_extrema_best.extremum.title()}Val')
    log(running_extrema_worst.extrema_dict, epoch_idx=epoch_idx,
        section=f'{running_extrema_worst.extremum.title()}Val')
