"""
INoD Dataset
author: Julia Hindel
"""

import albumentations as A
from PIL import Image
import numpy as np
import torch
import pickle
import os


class LoadImage:

    def __init__(self, RGB=True):
        self.RGB = RGB

    def __call__(self, path):
        with open(path, "rb") as f:
            rgb = Image.open(f).convert("RGB")
            rgb = np.array(rgb)
            if not self.RGB:
                rgb = rgb[:, :, ::-1]
            return rgb


def get_filelist(path, dataset_split):
    """
    Get a list of all files from ImageNet-like .txt file
    :param path: path to image folder
    :param dataset_split: name of txt file without file ending
    :return:
    """
    with open(f'{path}/{dataset_split}.txt', "r") as f:
        loaded_list = [line.rstrip() for line in f]
    loaded_tensor = np.asarray(loaded_list)
    return loaded_tensor


def get_norm_from_file(path):
    """
    Load norm from pickle files
    :param path: path to mean and std files
    :param nir: boolean; return mean and std of nir channel
    :return:
    """
    mean_file = os.path.join(path, 'mean')
    with open(mean_file, "rb") as fp:
        mean = pickle.load(fp) / 255.0

    std_file = os.path.join(path, 'std')
    with open(std_file, "rb") as fp:
        std = pickle.load(fp) / 255.0

    mean = mean[0:3]
    std = std[0:3]

    return A.Normalize(mean=mean, std=std)


class ConcatDataset(torch.utils.data.Dataset):
    """ Dataset class """

    def __init__(self, source_path, noise_path, crop_size,
                 random_noise=False, RGB=True, noise_split='train'):
        """
        init
        :param source_path: path of source files
        :param noise_path: path of noise files
        :param crop_size: image size to crop to during augmentations
        :param random_noise: boolean; retrieve artificial noise instead of images
        :param RGB: boolean; if RGB or BGR
        :param noise_split: name of dataset split to use.
        """
        super().__init__()

        self.source_filelist = get_filelist(source_path, 'train')
        self.source_path = source_path
        if not random_noise:
            self.noise_filelist = get_filelist(noise_path, noise_split)
            self.noise_path = noise_path

        # init methods
        self.read_image = LoadImage(RGB)

        # get normalization function
        normalize_source = normalize_noise = get_norm_from_file(source_path)

        # if random noise, replace noise function and normalization
        if random_noise:
            self.get_random_noise = self.uniform_noise
        else:
            # set noise function to default
            self.get_random_noise = self.get_noise_image

        # MoCo v2's augmentations
        scale = (1., 4.)
        if crop_size > 224:
            scale = (1., 2.)
        aug_list = [
            A.RandomResizedCrop(crop_size, crop_size, scale=scale),
            A.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
            A.ToGray(p=0.2),
            A.GaussianBlur(sigma_limit=[.1, 2.], p=0.5),
            A.HorizontalFlip(p=0.5)]

        augmentation_source = A.Compose(aug_list)
        self.transform_source = Transform(augmentation_source, normalize_source)

        aug_list_noise = aug_list.copy()
        noise_scale = (1., 4.)
        if ("syn_sb16" in noise_path) and (crop_size > 224):
            noise_scale = (.5, 1.)
            print("syn_sb16", noise_scale)
        aug_list_noise[0] = A.RandomResizedCrop(crop_size, crop_size, scale=noise_scale)

        augmentation_noise = A.Compose(aug_list_noise)
        self.transform_noise = Transform(augmentation_noise, normalize_noise)

    def __len__(self):
        return len(self.source_filelist)

    def __getitem__(self, i):
        # get source image
        source_file = self.source_filelist[i]
        source_aug = self.transform_source(self.read_image(f'{self.source_path}/train/{source_file}'))
        noise_aug = self.transform_noise(self.get_random_noise())
        return source_aug, noise_aug

    def get_noise_image(self):
        # get random noise image
        j = torch.randint(len(self.noise_filelist), (1,))
        noise_file = self.noise_filelist[j]
        noise = self.read_image(f'{self.noise_path}/train/{noise_file}')
        return noise

    def uniform_noise(self):
        # create artificial noise of size (800, 800, channel)
        return (np.random.randn(800, 800, 3) * np.asarray([19.63538837, 17.01363055, 12.08714193])
                + np.asarray([54.00258385, 46.92195182, 35.53070003])).astype("uint8")


class Transform:
    """Apply Transform to RGB and NIR."""

    def __init__(self, augmentation, normalization):
        """
        :param augmentation: augmentations to apply
        :param normalization: normalization to apply
        """
        # copy parameters
        self.augmentation = augmentation
        self.normalization = normalization

    def __call__(self, img):
        # augment all channels
        transformed = self.augmentation(image=img)
        rgb_t = transformed['image']
        # normalize complete image and transform to tensor
        img_norm = self.normalization(image=rgb_t)['image']
        img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1))
        return img_tensor
