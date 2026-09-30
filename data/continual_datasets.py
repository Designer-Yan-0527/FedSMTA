# ------------------------------------------
# Continual learning datasets used by this project.
# Only ImageNet-R is kept: the project runs cifar100 + ImageNet-R only.
# ------------------------------------------
import os

from shutil import move, rmtree
from typing import Any, Tuple

import numpy as np
import torch
from torchvision import datasets
from torchvision.datasets.utils import download_url

from PIL import Image
from tqdm import tqdm


class Imagenet_R(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        self.url = 'https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar'
        self.filename = 'imagenet-r.tar'

        self.fpath = os.path.join(root, 'imagenet-r')

        # BUGFIX: used to check the *extracted folder* with os.path.isfile(),
        # which is always False, so it re-entered the download branch on every
        # construction. Check the tar file instead.
        tar_path = os.path.join(root, self.filename)
        if not os.path.isfile(tar_path):
            if not download:
                raise RuntimeError('Dataset not found. You can use download=True to download it')
            else:
                print('Downloading from ' + self.url)
                download_url(self.url, root, filename=self.filename)

        if not os.path.exists(self.fpath):
            import tarfile
            tar_ref = tarfile.open(tar_path, 'r')
            tar_ref.extractall(root)
            tar_ref.close()

        # first construction only: 80/20 split, then move files into
        # imagenet-r/train and imagenet-r/test
        if not os.path.exists(self.fpath + '/train') and not os.path.exists(self.fpath + '/test'):
            self.dataset = datasets.ImageFolder(self.fpath, transform=transform)

            train_size = int(0.8 * len(self.dataset))
            val_size = len(self.dataset) - train_size

            train, val = torch.utils.data.random_split(self.dataset, [train_size, val_size])
            train_idx, val_idx = train.indices, val.indices

            self.train_file_list = [self.dataset.imgs[i][0] for i in train_idx]
            self.test_file_list = [self.dataset.imgs[i][0] for i in val_idx]
            self.split()

        if self.train:
            fpath = self.fpath + '/train'

        else:
            fpath = self.fpath + '/test'

        X, Y = [], []

        # BUGFIX: os.listdir() order is arbitrary on Linux (ext4), which would
        # silently misalign the class-index mapping between the train split and
        # the test/surrogate split. sorted() makes the mapping deterministic
        # and identical across instantiations (also required so that saved
        # data_split_indices point to the same images after a resume).
        folders = sorted(d for d in os.listdir(fpath)
                         if os.path.isdir(os.path.join(fpath, d)))

        for idx, folder in enumerate(tqdm(folders, desc=f'loading imagenet-r ({"train" if self.train else "test"})')):
            folder_path = os.path.join(fpath, folder)
            for ims in sorted(os.listdir(folder_path)):
                img_path = os.path.join(folder_path, ims)

                a = np.array(Image.open(img_path).convert('RGB'))

                X.append(a)
                Y.append(idx)

        self.data = X
        self.targets = Y

    def split(self):
        train_folder = self.fpath + '/train'
        test_folder = self.fpath + '/test'

        if os.path.exists(train_folder):
            rmtree(train_folder)
        if os.path.exists(test_folder):
            rmtree(test_folder)
        os.mkdir(train_folder)
        os.mkdir(test_folder)

        for c in self.dataset.classes:
            if not os.path.exists(os.path.join(train_folder, c)):
                os.mkdir(os.path.join(os.path.join(train_folder, c)))
            if not os.path.exists(os.path.join(test_folder, c)):
                os.mkdir(os.path.join(os.path.join(test_folder, c)))

        for path in self.train_file_list:
            if '\\' in path:
                path = path.replace('\\', '/')
            src = path
            dst = os.path.join(train_folder, '/'.join(path.split('/')[-2:]))
            move(src, dst)

        for path in self.test_file_list:
            if '\\' in path:
                path = path.replace('\\', '/')
            src = path
            dst = os.path.join(test_folder, '/'.join(path.split('/')[-2:]))
            move(src, dst)

        for c in self.dataset.classes:
            path = os.path.join(self.fpath, c)
            rmtree(path)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        img, target = self.data[index], int(self.targets[index])
        try:
            img = Image.fromarray(img)
        except:
            pass
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)
