import numpy as np
import torch
import torch.nn.functional as F
import random


def get_rot_mat(theta, device):
    theta = torch.tensor(theta)
    return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                         [torch.sin(theta), torch.cos(theta), 0]], device=device)


def rot_img(x, theta, dtype, device):
    rot_mat = get_rot_mat(theta, device)[None, ...].type(dtype).repeat(x.shape[0],1,1)
    grid = F.affine_grid(rot_mat, x.size(), align_corners=False).type(dtype).to(device)
    x = F.grid_sample(x, grid, align_corners=False).to(device)
    return x


def augmentation(images, labels, mask, device):
    prob_rot = np.random.uniform(0, 1)
    if prob_rot < 0.3:
        return images, labels, mask
    else:
        theta_list = [0., np.pi/2, np.pi, -np.pi/2]
        theta = random.choice(theta_list)

        dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        rotated_im = torch.zeros(images.size())
        rotated_labels_pre = torch.zeros(labels.size())
        rotated_labels = torch.zeros(labels.size())
        rotated_mask = torch.zeros(mask.size())

        for frame in range(images.shape[4]):

            rotated_im[:, :, :, :, frame] = rot_img(images[:, :, :, :, frame], theta, dtype, device)
            rotated_labels_pre[:, :, :, :, frame] = rot_img(labels[:, :, :, :, frame], theta, dtype, device)
            rotated_mask[:, :, :, :, frame] = rot_img(mask[:, :, :, :, frame], theta, dtype, device)

            u_labels = rotated_labels_pre[:, 0, :, :, frame]
            v_labels = rotated_labels_pre[:, 1, :, :, frame]
            rotated_labels[:, 0, :, :, frame] = u_labels * torch.cos(torch.tensor(theta)) + v_labels * torch.sin(torch.tensor(theta))
            rotated_labels[:, 1, :, :, frame] = -u_labels * torch.sin(torch.tensor(theta)) + v_labels * torch.cos(torch.tensor(theta))

        return rotated_im.to(device), rotated_labels.to(device), rotated_mask.to(device)