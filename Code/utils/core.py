from PIL import Image
import numpy as np
import torch
import ipdb
from torch.utils import data

def minmaxscaler(data, step):
    data = torch.where(data < 0, torch.full_like(data, 0), torch.full_like(data, step))
    # data = torch.where((data > 0) & (data < data.mean()),
    #                    torch.full_like(data, step), torch.full_like(data, step*2))
    # data = (data - torch.mean(data)) / (torch.std(data)+1e-5)
    # min_data = torch.min(data)
    # max_data = torch.max(data)
    # minmaxscaler = (data - min_data)*0.5/(max_data-min_data)
    return data


def poolingscaler(data, size, step):
    # [1, 3, 512, 512]
    # ipdb.set_trace()
    data = torch.nn.AvgPool2d((size, size), stride=size)(data.sign())*step
    data = torch.where(data < 0, torch.full_like(data, 0), data)
    data = torch.nn.functional.interpolate(data, scale_factor=size, mode='bilinear')
    
    return data


def val_transform(img_dir, crop_size, mean, label):
    # if not label:
    ori_image = Image.open(img_dir).convert('RGB')
    image = ori_image.resize(crop_size, Image.BICUBIC)
    image = np.asarray(image, np.float32)
    image = image[:, :, ::-1]  # change to BGR
    image -= mean
    image = image.transpose((2, 0, 1))
    # else:
    #     ori_image = Image.open(img_dir)
    #     image = ori_image.resize(crop_size, Image.NEAREST)
    #     image = np.asarray(image, np.float32)
    return torch.from_numpy(image.copy()).unsqueeze(0).type(torch.FloatTensor)

def norm(image):
    image_mean = torch.mean(image, axis=(2, 3), keepdims=True)
    image_std = torch.std(image, axis=(2, 3), keepdims=True)
    image = (image - image_mean) / image_std
    return image


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adain(content_features, style_features):
    content_mean, content_std = calc_mean_std(content_features)
    style_mean, style_std = calc_mean_std(style_features)
    normalized_features = (content_features - content_mean) + style_mean
    normalized_features = style_std * \
        (content_features - content_mean) / content_std + style_mean

    return normalized_features


def wct_style_transfer(encoder, decoder, content, style, eps):

    if content.shape[1]==1:
        content = torch.cat([content, content, content], dim=1)
    
    content_feat, content_skips = content, {}
    style_feat, style_skips = style, {}
    for level in [1, 2, 3, 4]:
        content_feat = encoder.encode(content_feat, content_skips, level)
        style_feat = encoder.encode(style_feat, style_skips, level)

    for level in [4, 3, 2]:
        content_feat = decoder.decode(content_feat, content_skips, level)
        style_feat = decoder.decode(style_feat, style_skips, level)

    fcs = adain(content_feat, style_feat)
    f_cs = decoder.decode(fcs, content_skips, 1)
    # ipdb.set_trace()
    content_gray = content[:,1,:,:].unsqueeze(1)
    out = eps*f_cs + (1-eps)*content_gray

    return torch.clamp(out, 0, 1)


def InfiniteSampler(n):
    # i = 0
    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0


class InfiniteSamplerWrapper(data.sampler.Sampler):
    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self):
        return 2 ** 31


# def adjust_learning_rate(lr_init, optimizer, iteration_count):
#     """Imitating the original implementation"""
#     lr = lr_init / (1.0 + 1e-3 * iteration_count)
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(learning_rate, optimizer, i_iter, num_steps):
    lr = lr_poly(learning_rate, i_iter, num_steps, 0.9)
    optimizer.param_groups[0]['lr'] = lr
    # if len(optimizer.param_groups) > 1:
    #     optimizer.param_groups[1]['lr'] = lr * 10


def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image