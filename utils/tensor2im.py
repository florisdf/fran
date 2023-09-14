from PIL import Image
import torch


def tensor2im(var):
    return Image.fromarray(
        (var.permute(1, 2, 0) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy(),
        'RGB'
    )
