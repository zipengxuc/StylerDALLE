import os
import torch
from torch.utils.data import Dataset
from typing import Tuple


class StylerDALLESLDataset(Dataset):

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        data = torch.load(os.path.join(self.root, self.files[item]))  # load the features of this sample
        token_32 = data["vtokens_32"].flatten()
        token_16 = data["vtokens_16"].flatten()
        image_name = data["image_name"]
        return token_16, token_32, image_name

    def __init__(self, data_path: str):
        self.root = data_path
        self.files = os.listdir(data_path)


class StylerDALLERLDataset(Dataset):

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        image_id = self.data[item]['image_id']
        caption = self.data[item]['caption'].lower().strip('.')
        prep_data = torch.load(os.path.join(self.root, 'coco_ViT-B_%s_%012d.pt' % (self.prefix, image_id)))
        token_16 = prep_data["vtokens_16"].flatten()
        token_32 = prep_data["vtokens_32"].flatten()
        image_32 = prep_data["images_32"]
        image_name = prep_data["image_name"]
        return caption, token_16, token_32, image_32, image_name

    def __init__(self, data, data_path):
        self.root = data_path
        self.data = data
        self.prefix = data_path.split('/')[-1].split('_')[0]
