import torch
from torchvision.transforms import v2
from torch.utils.data import DataLoader, Subset, Dataset

import numpy as np
import os
from PIL import Image


# https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip
class PennFudanDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms

        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):

        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        mask = np.array(mask)
        obj_ids = np.unique(mask)[1:]  # excluir el fondo

        boxes = []
        for obj_id in obj_ids:
            pos = np.where(mask == obj_id)
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones(
            (len(obj_ids),), dtype=torch.long
        )  # tantos 1 como boxes tenga (1 persona)

        if boxes.numel() == 0:
            boxes = boxes.reshape(0, 4)

        target = {"boxes": boxes, "labels": labels}

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target


def collate_fn(batch):
    return tuple(zip(*batch))  # hacer pares del subset images, labels


def get_pennfudan_dataloaders(batch_size=2, root="PennFudanPed"):
    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = PennFudanDataset(root=root, transforms=transform)

    train_indices = list(range(100))
    test_indices = list(range(100, len(dataset)))

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    category_names = ["background", "person"]

    return train_loader, test_loader, category_names
