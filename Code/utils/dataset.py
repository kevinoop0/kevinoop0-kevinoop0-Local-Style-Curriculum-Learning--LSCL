from torch.utils.data import Dataset
from PIL import Image
import torch
import glob
import numpy as np
import random
import ipdb
from torchvision import transforms
from seg.seg_utils import get_one_hot
import cv2
from utils.core import norm
from albumentations import Compose, VerticalFlip,HorizontalFlip, RandomSizedCrop, RandomScale, PadIfNeeded, RandomBrightnessContrast, RandomGamma, Rotate, GaussNoise
import os

def load_image(image_path, image_size, label=False):
    image = cv2.imread(image_path, 0)
    if not label:
        ori_image = cv2.resize(image, (image_size, image_size),
                           interpolation=cv2.INTER_CUBIC)
        ori_image = transforms.ToTensor()(ori_image)
        # ori_image = torch.cat([ori_image, ori_image, ori_image], dim=0)
        norm_img = norm(ori_image.unsqueeze(0))
        
        return ori_image, norm_img.squeeze(0)
    else:
        image = cv2.resize(image, (image_size, image_size),
                           interpolation=cv2.INTER_NEAREST)
        image = np.float32(image)
        image /= 85.
        image = transforms.ToTensor()(image)
        # image = get_one_hot(image.long())  # one-hot label
        return image


# def load_image_aug(image_path, label_path, image_size, label=False):
#     image = cv2.imread(image_path, 0)
#     label = cv2.imread(label_path, 0)
#     original_height, original_width = image.shape[0], image.shape[1]
#     # ipdb.set_trace()
#     aug = Compose([
#     RandomScale((-0.2, 0.2), p=1),
#     PadIfNeeded(min_height=original_height, min_width=original_width, value=0, border_mode=cv2.BORDER_CONSTANT),
#     HorizontalFlip(p=0.5),
#     VerticalFlip(p=0.5),
#     RandomBrightnessContrast(p=0.5),
#     RandomGamma(p=0.5),
#     Rotate(p=0.5, interpolation=cv2.INTER_NEAREST),
#     ])
#     augmented = aug(image=image, mask=label)
#     image_scaled = augmented['image']
#     mask_scaled = augmented['mask']

#     ori_image = cv2.resize(image_scaled, (image_size, image_size),interpolation=cv2.INTER_CUBIC)
#     ori_image = transforms.ToTensor()(ori_image)
#     norm_img = norm(ori_image.unsqueeze(0))
    
#     lab = cv2.resize(mask_scaled, (image_size, image_size),interpolation=cv2.INTER_NEAREST)
#     lab = np.float32(lab)
#     lab /= 85.
#     lab = transforms.ToTensor()(lab)
#     lab = get_one_hot(lab.long())  # one-hot label
#     return ori_image, norm_img.squeeze(0), lab


# class PairDataset_aug(Dataset):
#     def __init__(self, content_dir, label_dir, image_size):
#         super(PairDataset_aug, self).__init__()
#         self.image_size = image_size
#         self.content_images = glob.glob((content_dir + '/*'))
#         self.label_images = glob.glob((label_dir + '/*'))

#     def __getitem__(self, index):
#         content_path = self.content_images[index]
#         label_path = self.label_images[index]
#         assert content_path.split('/')[-1] == label_path.split('/')[-1]
#         origin_image, image, label = load_image_aug(content_path, label_path, self.image_size)
#         return {'origin':origin_image, 'image': image, 'label': label, 'name': content_path}

#     def __len__(self):
#         return len(self.content_images)

class PairDataset(Dataset):
    def __init__(self, content_dir, label_dir, image_size):
        super(PairDataset, self).__init__()
        self.image_size = image_size
        self.content_images = glob.glob((content_dir + '/*'))
        self.label_images = glob.glob((label_dir + '/*'))

    def __getitem__(self, index):
        content_path = self.content_images[index]
        label_path = self.label_images[index]
        assert content_path.split('/')[-1] == label_path.split('/')[-1]
        origin_image, norm_img = load_image(content_path, self.image_size)
        label = load_image(label_path, self.image_size, label=True)
        return {'ori_img':origin_image, 'norm_img':norm_img, 'label': label, 'name': content_path}

    def __len__(self):
        return len(self.content_images)


class StyleDataset(Dataset):
    def __init__(self, style_dir, image_size):
        super(StyleDataset, self).__init__()
        self.image_size = image_size
        self.style_images = glob.glob((style_dir + '/*'))

    def __getitem__(self, index):
        style_path = self.style_images[index]
        style = cv2.imread(style_path, 0)
        style = cv2.cvtColor(style, cv2.COLOR_GRAY2RGB) 
        style = cv2.resize(style, (self.image_size, self.image_size),
                           interpolation=cv2.INTER_CUBIC)
        style = transforms.ToTensor()(style)
        # ipdb.set_trace()
        name = os.path.basename(style_path)
        return {'image':style, 'name':name}

    def __len__(self):
        return len(self.style_images)


def auto_aug(image, label, h, w):
    aug = Compose([
    RandomScale((-0.1, 0.1), p=1),
    PadIfNeeded(min_height=h, min_width=w, value=0, border_mode=cv2.BORDER_CONSTANT),
    HorizontalFlip(p=0.5),
    VerticalFlip(p=0.5),
    RandomBrightnessContrast(p=0.5),
    RandomGamma(p=0.5),
    GaussNoise(p=0.5),
    Rotate(limit=45,p=0.5, value=0, border_mode=cv2.BORDER_CONSTANT),
    ])
    augmented = aug(image=image, mask=label)
    image_scaled = augmented['image']
    mask_scaled = augmented['mask']
    return image_scaled, mask_scaled

def img_process(image, label, image_size):
    image = cv2.resize(image, (image_size, image_size),interpolation=cv2.INTER_CUBIC)
    ori_image = transforms.ToTensor()(image)
    # ipdb.set_trace()
    # ori_image = torch.cat([ori_image, ori_image, ori_image], dim=0)
    norm_img = norm(ori_image.unsqueeze(0))
    
    lab = cv2.resize(label, (image_size, image_size),interpolation=cv2.INTER_NEAREST)
    lab = np.float32(lab)
    lab /= 85.
    lab = transforms.ToTensor()(lab)
    lab = get_one_hot(lab.long())
    return ori_image, norm_img.squeeze(0), lab


def load_image_aug(image_path, label_path, image_size):
    image = cv2.imread(image_path, 0)
    label = cv2.imread(label_path, 0)
    h, w = image.shape[0], image.shape[1]
    image_scaled, mask_scaled = auto_aug(image, label, h ,w)
    ori_img, norm_img, ori_lab = img_process(image, label, image_size)
    _, image_scaled, mask_scaled = img_process(image_scaled, mask_scaled, image_size)
    
    return ori_img, norm_img, ori_lab, image_scaled, mask_scaled


class PairDataset_aug(Dataset):
    def __init__(self, content_dir, label_dir, image_size):
        super(PairDataset_aug, self).__init__()
        self.image_size = image_size
        self.content_images = glob.glob((content_dir + '/*'))
        self.label_images = glob.glob((label_dir + '/*'))

    def __getitem__(self, index):
        content_path = self.content_images[index]
        label_path = self.label_images[index]
        # ipdb.set_trace()
        assert content_path.split('/')[-1] == label_path.split('/')[-1]
        ori_img, norm_img, ori_lab, image_scaled, mask_scaled = load_image_aug(content_path, label_path, self.image_size)
        return {'ori_img':ori_img, 'norm_img':norm_img, 'ori_lab': ori_lab, 'scale_img': image_scaled, 'scale_lab': mask_scaled}

    def __len__(self):
        return len(self.content_images)

