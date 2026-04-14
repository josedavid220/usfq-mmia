from torch.utils.data import Dataset, DataLoader 
from torchvision.transforms import v2
from PIL import Image 
import torch 
from glob import glob 

DATASET_PATH = "./flood_area_dataset"
DEFAULT_TRANSFORM = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

class FloodDataset(Dataset):
    def __init__(self, dataset_path=DATASET_PATH, transform=DEFAULT_TRANSFORM) -> None:
        self.img_names = []
        self.mask_names = []

        with open(f"{dataset_path}/metadata.csv", 'r', encoding='utf-8') as f:
            next(f)
            for line in f:
                line = line.strip()
                img_name, mask_name = line.split(",")

                img_name = f"{dataset_path}/Image/{img_name}"
                mask_name = f"{dataset_path}/Mask/{mask_name}"

                self.img_names.append(img_name)
                self.mask_names.append(mask_name)
            
        self.transform = transform 

    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, idx):
        img_string = self.img_names[idx]
        mask_string = self.mask_names[idx]

        img = Image.open(img_string).convert("RGB")
        img.load()
        img = self.transform(img)

        mask = Image.open(mask_string).convert("RGB")
        mask.load()
        mask = self.transform(mask)
        mask = v2.Grayscale(1)(mask)

        return img, mask 

def load_data(dataset_path=DATASET_PATH, transform=DEFAULT_TRANSFORM, num_workers=0,
              batch_size=128):
    dataset = FloodDataset(dataset_path, transform)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size,
                      shuffle=True, drop_last=False)