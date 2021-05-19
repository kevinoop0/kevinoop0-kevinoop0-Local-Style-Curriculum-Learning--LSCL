from torchvision import transforms
from PIL import Image
import torch.utils.data as data
# from pathlib import Path
import cv2
import torch
import numpy as np
import torch.nn.functional as F
import ipdb
from scipy.ndimage import measurements
import copy

def remove_minor_cc(vol_data, rej_ratio=0.3, rename_map=[1,2,3]):
    """Remove small connected components refer to rejection ratio"""

    rem_vol = copy.deepcopy(vol_data)
    class_n = len(rename_map)
    # retrieve all classes
    for c in range(1, class_n+1):
        # print 'processing class %d...' % c
        class_idx = (vol_data==rename_map[c-1])*1
        class_vol = np.sum(class_idx)
        labeled_cc, num_cc = measurements.label(class_idx)
        # retrieve all connected components in this class
        for cc in range(1, num_cc+1):
            single_cc = ((labeled_cc==cc)*1)
            single_vol = np.sum(single_cc)
            # remove if too small
            if single_vol / (class_vol*1.0) < rej_ratio:
                rem_vol[labeled_cc==cc] = 0

    return rem_vol



class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val=0, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

def rotate_img(img, angle=90):
    h, w = img.shape[:2]
    RotateMat = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    RotateImg = cv2.warpAffine(img, RotateMat, (w, h), flags=cv2.INTER_CUBIC)
    return RotateImg

def aug_rotate(img, angle):
    
    arr_img = img.squeeze(1).cpu().detach().numpy()*255.0
    all_rot_img = torch.zeros_like(img)
    for i in range(arr_img.shape[0]):
        rot_img = rotate_img(arr_img[i])
        tensor_img = torch.from_numpy(rot_img/255.0).unsqueeze(0)
        all_rot_img[i] = tensor_img
    
    return all_rot_img
    

# def get_mask(mask):
#     contours, hierarchy = cv2.findContours(mask, 1, 2)
#     num = len(contours)
#     areas = []
#     for i in range(0, num):
#         cnt = contours[i]
#         area = cv2.contourArea(cnt)
#         areas.append(area)
#     if areas:
#         areas_tar = sorted(areas)[:-1]  # 要去掉的洞洞
#         for idx in areas_tar:
#             id = areas.index(idx)
#             cntMax = contours[id]
#             cv2.drawContours(mask, [cntMax], 0, 0, -1)  # -1是填充模式  第一个0是轮廓厚度
#         return mask
#     else:
#         return mask

def get_mask(mask, image_size):
    contours, hierarchy = cv2.findContours(mask, 1, 2)
    num = len(contours)
    areas = []
    for i in range(0, num):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        areas.append(area)
    maskFillContour = np.zeros((image_size, image_size), np.uint8)

    if areas:

        id = areas.index(max(areas))
        cntMax = contours[id]
        cv2.drawContours(maskFillContour, [cntMax], 0, 255, -1) # -1 是填充模式

    return maskFillContour


def get_one_hot(label, num_classes=4):
    size = list(label.size())
    label = label.view(-1)
    ones = torch.sparse.torch.eye(num_classes)
    ones = ones.index_select(0, label)
    size.append(num_classes)
    label = ones.view(*size[1:])
    return label.permute(2, 0, 1)


def get_one_hot_batch(label, num_classes=4):  # b x h x w
    device = label.device
    size = list(label.size())
    label = label.view(-1)
    ones = torch.sparse.torch.eye(num_classes, device=device)
    ones = ones.index_select(0, label)
    size.append(num_classes)
    label = ones.view(*size)
    return label.permute(0, 3, 1, 2)


def compute_dice(pred, label):
    intersection = np.sum(pred * label)
    union = np.sum(pred) + np.sum(label)
    return 2. * intersection / float(union + 0.0001)


def get_dice_ob(pred, mask):
    pred = pred.numpy()
    pred = np.argmax(pred, 1)
    # ipdb.set_trace()
    mask = mask.numpy()
    mask = np.argmax(mask, 1)
    class_id = np.max(mask)

    if class_id == 1:
        pred = pred / 1.
    elif class_id == 2:
        pred = pred / 2.
    elif class_id == 3:
        pred = pred / 3.
    pred = np.reshape(pred, (-1, ))
    mask = np.reshape(mask, (-1, ))
    index = np.arange(len(pred))
    falseIndex = index[(pred != 0) & (pred != 1)]
    for i in falseIndex:
        if mask[i] == 0:
            pred[i] = 1
        else:
            pred[i] = 0
    dice = compute_dice(pred, mask)
    return dice


def focal_loss(inputs, targets):
    gamma = 2.
    alpha = 0.25
    class_num = 4
    p1 = torch.where(targets == 1, inputs, torch.ones_like(inputs))
    p0 = torch.where(targets == 0, inputs, torch.zeros_like(inputs))
    p1 = torch.clamp(p1, 1e-6, .999999)
    p0 = torch.clamp(p0, 1e-6, .999999)
    return (-torch.sum((1 - alpha) * torch.log(p1)) - torch.sum(alpha * torch.pow(p0, gamma) * torch.log(torch.ones_like(p0) - p0))) / float(class_num)


def dice_loss(inputs, targets):
    smooth = 1.
    iflaten = inputs.reshape(-1)
    tflaten = targets.reshape(-1)
    intersection = (iflaten * tflaten).sum()
    return 1 - ((2. * intersection + smooth) / (iflaten.sum() + tflaten.sum() + smooth))


class DiceFocalLoss(torch.nn.Module):
    def __init__(self):
        super(DiceFocalLoss, self).__init__()

    def forward(self, inputs, targets):
        inputs = torch.softmax(inputs, dim=1)
        beta = 0.0001
        return beta * focal_loss(inputs, targets) + 10*dice_loss(inputs, targets)

class DiceLoss(torch.nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets):
        inputs = torch.softmax(inputs, dim=1)
        beta = 0.0
        return beta * focal_loss(inputs, targets) + (1-beta)*dice_loss(inputs, targets)


def CrossE_loss(pred, label, nclass=4):
    label = label.long().cuda()
    ce_loss = torch.nn.CrossEntropyLoss().cuda()
    dice_loss = DiceLoss().cuda()
    pred = pred.permute(0, 2, 3, 1).contiguous()
    pred = pred.view(-1, nclass)
    label = label.contiguous().view(-1)
    # ipdb.set_trace()
    return ce_loss(pred, label)

