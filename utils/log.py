from typing import Any, Dict, Optional

import torch
import wandb


def log(
    log_dict: Dict[str, Any],
    epoch_idx: int,
    batch_idx: Optional[int] = None,
    section: Optional[int] = None
):
    """Logs the given dict to WandB.

    Args:
        log_dict: The dictionary to log.
        epoch_idx: The epoch number.
        batch_idx: The batch number.
        section: The section to put the logs in.
    """
    def get_key(k):
        if section is None:
            return k
        else:
            return f'{section}/{k}'

    def get_value(v):
        if isinstance(v, torch.Tensor):
            return v.detach().cpu()
        elif isinstance(v, float) or isinstance(v, int):
            return v
        else:
            return None

    for k, v in log_dict.items():
        k = get_key(k)
        v = get_value(v)
        if v is None:
            continue
        wandb_dict = {k: v, "epoch": epoch_idx}
        if batch_idx is not None:
            wandb_dict['batch_idx'] = batch_idx
        wandb.log(wandb_dict)
