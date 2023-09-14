from PIL import Image
from torchvision.transforms import Compose, ToTensor, Resize, Normalize


test_transform = Compose([
    Resize(size=(256, 256), interpolation=Image.BILINEAR),
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
