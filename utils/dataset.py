import os
from torch.utils.data import Dataset
from PIL import Image

class HorseZebraDataset(Dataset):
    def __init__(self, root, transform=None, mode='train'):
        self.root = root
        self.transform = transform
        self.mode = mode
        self.A_path = os.path.join(root, f'{mode}A')
        self.B_path = os.path.join(root, f'{mode}B')
        self.A_images = os.listdir(self.A_path)
        self.B_images = os.listdir(self.B_path)

    def __len__(self):
        return max(len(self.A_images), len(self.B_images))

    def __getitem__(self, index):
        A_img = Image.open(os.path.join(self.A_path, self.A_images[index % len(self.A_images)])).convert('RGB')
        B_img = Image.open(os.path.join(self.B_path, self.B_images[index % len(self.B_images)])).convert('RGB')

        if self.transform:
            A_img = self.transform(A_img)
            B_img = self.transform(B_img)

        return {'A': A_img, 'B': B_img}
