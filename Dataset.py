from torch.utils.data import Dataset
from PIL import Image


class LM(Dataset):

    def __init__(self, X, Y, transform=None):
        self.path = X
        self.Y = Y
        self.transform = transform

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        path = self.path[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = int(self.Y[idx])

        return image, label
