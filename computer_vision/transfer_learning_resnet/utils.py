from sklearn.model_selection import train_test_split

import os 
from glob import glob 

DATASET_PATH = './music_instruments'

def split_train_val(dataset_origen: str = DATASET_PATH):
    y = [] # nombre de la clase a la que pertenecen
    X = [] # nombre del archivo de la imagen

    class_names = []
    clases = os.listdir(dataset_origen) # nombres de las carpetas
    
    for clase in clases:
        if ".csv" in clase:
            clases.remove(clase)

    for class_id, clase in enumerate(clases):
        class_names.append(clase)
        file_names = glob(f"{dataset_origen}/{clase}/*.*")
        for file_name in file_names:
            X.append(file_name) # accordion/0001.jpg
            y.append(class_id) # 0 -> accordion
    
    # 10 instrumentos * 200 aprox
    # Relativamente Balanceado (% mismo porcentaje de participacion de cada clase)
    # Corte Stratify (asegurarse que haya participacion de cada clase con el porcentaje acordado tanto en
    # el train como el test)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    with open("./class_meaning.csv", 'w', encoding='utf-8') as f:
        for class_id, class_name in enumerate(class_names):
            f.write(f"{class_id};{class_name}\n")
    
    with open("./train_dataset.csv", 'w', encoding='utf-8') as f:
        for (filename, class_) in zip(X_train, y_train):
            f.write(f"{filename};{class_}\n")
    
    with open("./test_dataset.csv", 'w', encoding='utf-8') as f:
        for (filename, class_) in zip(X_test, y_test):
            f.write(f"{filename};{class_}\n")

from torch.utils.data import Dataset, DataLoader, RandomSampler
from torchvision import transforms
from PIL import Image 
import torch 

class InstrumentsDataset(Dataset):
    def __init__(self, dataset_csv: str,
                 transform=transforms.ToTensor()):
        
        self.img_names = []
        self.class_idx = []
        self.transform = transform

        with open(dataset_csv, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                img_name, class_id = line.split(";")
                self.img_names.append(img_name)
                self.class_idx.append(int(class_id))
    
    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        # JIT Just In Time # ((N, 3, H, W), [target0, target1, ...])
        img_names = self.img_names[idx]
        img = Image.open(img_names).convert('RGB')

        img.load()

        img = self.transform(img)
        class_id = torch.tensor(self.class_idx[idx], dtype=torch.int64)

        return img, class_id
    
def load_data(dataset_path: str, transform=transforms.ToTensor(),
              num_workers: int = 0, batch_size=128, shuffle: bool = True):
            
    dataset = InstrumentsDataset(dataset_path, transform)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size,
                        shuffle=shuffle,
                        drop_last=False)