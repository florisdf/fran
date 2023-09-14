from pathlib import Path

import torch
from torch import nn


def create_checkpoints(
    fran: nn.Module,
    patchGAN: nn.Module,
    run_name: str,
    ckpt_dir: Path,
    save_best: bool = False,
    save_last: bool = False,
):
    """Creates checkpoints for the model, labeled as 'best' and/or 'last'.

    For both ``save_best`` and ``save_last``, a separate checkpoint is created.
    If neither ``save_best`` or ``save_last`` is ``True``, no checkpoint is
    created. If either is ``True``, only a single checkpoint is created.

    Args:
        fran: FRAN model to create a checkpoint for.
        patchGAN: PatchGAN model to create a checkpoint for.
        run_name: A name to prefix the checkpoint filename with.
        ckpt_dir: The directory to store the checkpoint in.
        save_best: Whether we should save this model as the 'best'.
        save_last: Whether we should save this model as the 'last'.
    """
    file_prefix = f"{run_name}_"
    file_suffix = '.pth'

    if not ckpt_dir.exists():
        ckpt_dir.mkdir(parents=True)

    fran_state_dict = fran.state_dict()
    patchGAN = patchGAN.state_dict()
    ckpt_dict = {
        'FRAN': fran_state_dict,
        'PatchGAN': patchGAN,
    }

    if save_best:
        torch.save(
            ckpt_dict,
            ckpt_dir
            / f'{file_prefix}best{file_suffix}'
        )

    if save_last:
        torch.save(
            ckpt_dict,
            ckpt_dir / f'{file_prefix}last{file_suffix}'
        )
