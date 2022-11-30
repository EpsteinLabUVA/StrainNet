import torch


def EPE_loss(output, target, mask):
    epe_tv = torch.sum(torch.sqrt(((output[:, 0, :, :, :] - target[:, 0, :, :, :])**2.0
                                 + (output[:, 1, :, :, :] - target[:, 1, :, :, :])**2.0)) * mask[:, 0, :, :, :])\
           / torch.sum(mask[:, 0, :, :, :]) \
           + 0.000 * torch.sum(torch.abs(output[:, :, :, :, :-1] - output[:, :, :, :, 1:]) * mask[:, :, :, :, 1:])\
           / torch.sum(mask[:, :, :, :, :])
    return epe_tv
