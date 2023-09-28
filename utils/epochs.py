from pathlib import Path
import sys

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
import wandb

from .log import log
from .ckpts import create_checkpoints

train_batch_idx = -1  # should have global scope


def run_epoch(
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

    discr_weight: float,

    epoch_idx: int,
    device: torch.device,
    dl_train: DataLoader,
    dl_val: DataLoader,
    discr_steps: int,

    save_last,
    ckpt_dir,
    run_name,

    val_every: int,
):
    global train_batch_idx

    for (train_batch_idx, batch) in enumerate(
        tqdm(dl_train, leave=False),
        start=train_batch_idx + 1,
    ):
        fran.train()
        discr.train()
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
        loss_fake = bce_loss_fn(pred_score_fake,
                                torch.zeros_like(pred_score_fake))

        # Losses for real images with true age
        pred_score_real1 = torch.sigmoid(discr(src_img, src_age))
        loss_real1 = bce_loss_fn(pred_score_real1,
                                 torch.ones_like(pred_score_real1))

        pred_score_real2 = torch.sigmoid(discr(tgt_img, tgt_age))
        loss_real2 = bce_loss_fn(pred_score_real2,
                                 torch.ones_like(pred_score_real2))

        # Also compute loss for real images with wrong ages; these should also
        # give a score of zero
        wrong_age = get_wrong_ages(src_age, dl_train.dataset.max_age,
                                   dl_train.dataset.min_age)
        pred_score_wrong = torch.sigmoid(discr(src_img, wrong_age))
        loss_wrong = bce_loss_fn(pred_score_wrong,
                                 torch.zeros_like(pred_score_wrong))

        # Sum losses
        discr_loss = (loss_fake + loss_real2 + loss_real2
                      + loss_wrong)*discr_weight

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

        if train_batch_idx % val_every == 0:
            # Validation epoch
            fran.eval()
            discr.eval()
            validation_epoch(
                fran=fran,
                discr=discr,

                epoch_idx=epoch_idx,
                device=device,
                dl_val=dl_val,

                save_last=save_last,
                ckpt_dir=ckpt_dir,
                run_name=run_name,
            )


def get_wrong_ages(true_ages, max_age, min_age):
    age_grid = torch.stack([torch.randperm(max_age - min_age + 1) + min_age
                            for _ in range(len(true_ages))]).type_as(true_ages)
    age_grid = age_grid[
        age_grid != true_ages[:, None]
    ].reshape((age_grid.shape[0], age_grid.shape[1] - 1))
    return age_grid[:, 0]


@torch.no_grad()
def validation_epoch(
    fran,
    discr,

    epoch_idx: int,
    device: torch.device,
    dl_val: DataLoader,

    save_last: bool,
    ckpt_dir: Path,
    run_name: str,

    img_log_size: tuple = (200, 100),
):
    wandb_ims = []

    for (val_batch_idx, batch) in enumerate(tqdm(dl_val, leave=False)):
        src_img, src_age, tgt_img, tgt_age = batch
        src_img = src_img.to(device)
        src_age = src_age.to(device)

        tgt_img = tgt_img.to(device)
        tgt_age = tgt_age.to(device)

        # Predict image with tgt age
        fran_img = fran(src_img, src_age, tgt_age)

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
            wandb.Image(to_pil_image(
                torch.cat([inv_norm(img1), inv_norm(img2)], dim=-1).cpu()
            ).resize(img_log_size),
                        caption=f'{age1} -> {age2}')
        )

    wandb.log({
        'Examples': wandb_ims,
        'batch_idx': train_batch_idx,
    })

    # Create checkpoints
    create_checkpoints(fran, discr, run_name, ckpt_dir,
                       save_best=False,
                       save_last=save_last)
