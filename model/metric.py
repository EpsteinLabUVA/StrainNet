import torch


def EPE(output, target, mask):
    epe = torch.sum(torch.sqrt(((output[:, 0, :, :, :] - target[:, 0, :, :, :]) ** 2.0 +
                                   (output[:, 1, :, :, :] - target[:, 1, :, :, :]) ** 2.0)) * mask[:, 0, :, :, :]) \
             / torch.sum(mask[:, 0, :, :, :])
    return epe


def EPE_TV(output, target, mask):
    epe_tv = torch.sum(torch.sqrt(((output[:, 0, :, :, :] - target[:, 0, :, :, :])**2.0 +
                                 (output[:, 1, :, :, :] - target[:, 1, :, :, :])**2.0)) * mask[:, 0, :, :, :])\
           / torch.sum(mask[:, 0, :, :, :]) \
           + 0.000 * torch.sum(torch.abs(output[:, :, :, :, :-1] - output[:, :, :, :, 1:]) * mask[:, :, :, :, 1:])\
           / torch.sum(mask[:, :, :, :, :])
    return epe_tv


def L1(output, target, mask):
    L1 = torch.sum(torch.abs((output - target) * mask)) / torch.sum(mask)
    return L1


def TV(output, mask):
    tv = 0.000 * torch.sum(torch.abs(output[:, :, :, :, :-1] - output[:, :, :, :, 1:]) * mask[:, :, :, :, 1:])\
           / torch.sum(mask[:, :, :, :, :])
    return tv