import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
    
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))  
        self.image_paths = []
        self.labels = []

        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            for image_name in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image_name)
                self.image_paths.append(image_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")  
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def get_data_loaders(train_dir, test_dir, batch_size=8):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = CustomImageDataset(root_dir=train_dir, transform=transform)
    test_dataset = CustomImageDataset(root_dir=test_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, train_dataset.classes

if __name__ == "__main__":
    train_loader, test_loader, classes = get_data_loaders("dataset/train", "dataset/test")
    print(f"Classes: {classes}")
    print(f"Number of training samples: {len(train_loader.dataset)}")
    print(f"Number of test samples: {len(test_loader.dataset)}")

    for images, labels in train_loader:
        print(f"Batch size: {images.size(0)}")
        print(f"Image shape: {images.shape}")
        print(f"Labels: {labels}")
        break
