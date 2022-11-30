from torchvision import datasets, transforms
from base import BaseDataLoader
from data_loader.dataset import Dataset_StrainNet


class StrainNetDataLoader(BaseDataLoader):
    """
    DTSA data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, is_cine, batch_size, shuffle=True, validation_split=0.0, num_workers=1):
        self.data_dir = data_dir
        self.dataset = Dataset_StrainNet(file_root=data_dir, is_cine=is_cine)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
