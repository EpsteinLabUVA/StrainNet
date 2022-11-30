import scipy.io as sio
from glob import glob
from os.path import *
import numpy as np
from torch.utils.data import Dataset
import torch


class Dataset_StrainNet(Dataset):

    def __init__(self, file_root, is_cine=False, transform=None):
        """Init function should not do any heavy lifting, but
            must initialize how many items are available in this data set.
        """
        self.transform = transform
        self.file_list = sorted(glob(join(file_root, '*')))
        self.data_len = len(self.file_list)
        self.is_cine = is_cine

    def __len__(self):
        """return number of points in our dataset"""

        return self.data_len

    def __getitem__(self, index):
        """ Here we have to return the item requested by `idx`
            The PyTorch DataLoader class will use this method to make an iterable for
            our training or validation loop.
        """
        if not self.is_cine:
            input_pre = sio.loadmat(join(self.file_list[index], 'input.mat'))
            label_pre = sio.loadmat(join(self.file_list[index], 'label.mat'))

            input_pre2 = np.float32(input_pre['input'])
            label_pre2 = np.float32(label_pre['label'])

            mask_pre = np.float32(label_pre2.astype(bool).astype(int))

            input = torch.tensor(input_pre2)
            label = torch.tensor(label_pre2)

            mask = torch.tensor(mask_pre)

            assert not np.any(np.isnan(np.float64(input_pre['input'])))

            return input, label, mask

        else:
            input_pre = sio.loadmat(join(self.file_list[index], 'input.mat'))

            input_pre2 = np.float32(input_pre['input'])

            mask_pre = np.float32(input_pre2.astype(bool).astype(int))

            input = torch.tensor(input_pre2)

            mask = torch.tensor(mask_pre)

            assert not np.any(np.isnan(np.float64(input_pre['input'])))

            return input, mask