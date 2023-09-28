from itertools import product
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from .kfold import k_fold_trainval_split


class FRANDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        is_val: bool,
        transform: Optional[Callable] = None,
        num_folds: int = 5,
        val_fold: int = -1,
        n_subsample: int = None,
        subsample_seed: int = 42,
    ):
        self.data_root = Path(data_root)
        self.transform = transform

        df_all = pd.DataFrame([
            {
                'label': p.parent.name,
                'age': int(p.stem.split('_')[-1]),
                'image': p,
            }
            for p in self.data_root.glob('*/*.jpg')
        ])
        self.min_age = df_all['age'].min()
        self.max_age = df_all['age'].max()

        df_train, df_val = k_fold_trainval_split(
            df_all, num_folds=num_folds, val_fold=val_fold
        )

        self.df = df_val if is_val else df_train

        if n_subsample is not None:
            self.df = self.df.sample(
                n_subsample, random_state=subsample_seed,
                ignore_index=True,
            )

        self.pairs = [
            (i, j)
            for label, group in self.df.groupby('label')
            for i, j in product(group.index, group.index)
        ]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        i, j = self.pairs[idx]

        img1 = self.df.loc[i, 'image']
        age1 = self.df.loc[i, 'age']

        img2 = self.df.loc[j, 'image']
        age2 = self.df.loc[j, 'age']

        im1 = Image.open(img1)
        im2 = Image.open(img2)

        if self.transform is not None:
            im1 = self.transform(im1)
            im2 = self.transform(im2)

        return (im1, age1, im2, age2)
