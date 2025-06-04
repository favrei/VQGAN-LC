import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader


class CenterDataset(Dataset):
    def __init__(self, directory: str):
        self.files = [os.path.join(directory, f) for f in os.listdir(directory)]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(np.load(self.files[idx]))


dir_path = "clustering_centers_random_100K_VitB"
dataset = CenterDataset(dir_path)
loader = DataLoader(dataset, batch_size=1, num_workers=4, pin_memory=True)

features = []
for batch in loader:
    features.append(batch.squeeze(0))

features = torch.cat(features, dim=0)
torch.save(features, "random_codebook_1000cls_100000_vitb.pth")
print(features.shape)
