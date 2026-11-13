import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, Subset
from sklearn.model_selection import train_test_split
import numpy as np
from src.config import image_folder_path , pretrained_batch_size


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3), 
    transforms.Resize((224, 224)),               
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD) 
])

test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD) 
])

class DatasetWrapper(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

raw_dataset = datasets.ImageFolder(root=image_folder_path)
targets = raw_dataset.targets


train_idx, test_idx = train_test_split(
    np.arange(len(targets)), 
    test_size=0.2, 
    stratify=targets, 
    random_state=42
)

train_subset = Subset(raw_dataset, train_idx)
test_subset = Subset(raw_dataset, test_idx)


train_dataset = DatasetWrapper(train_subset, transform=train_transform)
test_dataset = DatasetWrapper(test_subset, transform=test_transform)


train_labels = [targets[i] for i in train_idx]

class_counts = np.bincount(train_labels)
class_weights = 1.0 / class_counts
sample_weights = [class_weights[label] for label in train_labels]

sampler = WeightedRandomSampler(
    weights=sample_weights, 
    num_samples=len(sample_weights), 
    replacement=True
)


train_loader = DataLoader(train_dataset, batch_size=pretrained_batch_size, sampler=sampler)
test_loader = DataLoader(test_dataset, batch_size=pretrained_batch_size, shuffle=False)

if __name__ == "__main__":
    print(f"Training batches: {len(train_loader)}")
    print(f"Testing batches: {len(test_loader)}")
    print(f"Classes: {raw_dataset.classes}")
    
    
    
   