import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

# Hyper Paramters
WEIGHT_DECAY = 0.0005
LEARNING_RATE = 2.5e-4
LEARNING_RATE_D = 1e-4
NUM_STEPS = 100000
NUM_STEPS_STOP = 100000  # Use damping instead of early stopping
WARMUP_STEPS = int(NUM_STEPS_STOP/20)
POWER = 0.9
RANDOM_SEED = 1234

class CrossEntropy2d(nn.Module):

    def __init__(self, class_num=19, alpha=None, gamma=2, size_average=True, ignore_label=255):
        super(CrossEntropy2d, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target):
        N, C, H, W = predict.size()
        sm = nn.Softmax(dim=0)
        predict = predict.transpose(0, 1).contiguous()
        P = sm(predict)
        P = torch.clamp(P, min=1e-9, max=1-(1e-9))

        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        predict = P[target_mask.view(
            1, N, H, W).repeat(C, 1, 1, 1)].view(C, -1)
        probs = torch.gather(predict, dim=0, index=target.view(1, -1))
        log_p = probs.log()
        batch_loss = -(torch.pow((1-probs), self.gamma))*log_p

        if self.size_average:
            loss = batch_loss.mean()
        else:

            loss = batch_loss.sum()
        return loss


def loss_calc(pred, label):
    label = Variable(label.long()).cuda()
    criterion = CrossEntropy2d(19).cuda()
    return criterion(pred, label)


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def lr_warmup(base_lr, iter, warmup_iter):
    return base_lr * (float(iter) / warmup_iter)


def adjust_learning_rate(optimizer, i_iter):
    if i_iter < WARMUP_STEPS:
        lr = lr_warmup(LEARNING_RATE, i_iter, WARMUP_STEPS)
    else:
        lr = lr_poly(LEARNING_RATE, i_iter, NUM_STEPS, POWER)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def adjust_learning_rate_D(optimizer, i_iter):
    if i_iter < WARMUP_STEPS:
        lr = lr_warmup(LEARNING_RATE_D, i_iter, WARMUP_STEPS)
    else:
        lr = lr_poly(LEARNING_RATE_D, i_iter, NUM_STEPS, POWER)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10
