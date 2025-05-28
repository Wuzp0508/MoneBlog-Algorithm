import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
from PIL import Image
import copy
from tqdm import tqdm
import logging
from datetime import datetime
from sklearn.metrics import f1_score
from resnet18 import ResNet18

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epoch_num = 100
save_interval = 10

if not os.path.exists('log'):
    os.makedirs('log')

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f'log/train_log_{current_time}.txt'
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(message)s')
logging.info("训练开始")

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_val = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class CustomImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        """
        Args:
            image_paths (list of str): 图片的路径列表。
            labels (list of int): 图片的标签。
            transform (callable, optional): 用于图片预处理的转换。
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.classes = sorted(list(set(self.labels)))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

def load_data(image_folder):
    image_paths = []
    labels = []
    for label, class_name in enumerate(os.listdir(image_folder)):
        class_folder = os.path.join(image_folder, class_name)
        if os.path.isdir(class_folder):
            for img_file in os.listdir(class_folder):
                img_path = os.path.join(class_folder, img_file)
                if img_path.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    image_paths.append(img_path)
                    labels.append(label)
    return image_paths, labels

train_image_paths, train_labels = load_data('./Animal_recognition/train')
val_image_paths, val_labels = load_data('./Animal_recognition/val')

train_dataset = CustomImageDataset(train_image_paths, train_labels, transform=transform_train)
val_dataset = CustomImageDataset(val_image_paths, val_labels, transform=transform_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

class_names = train_dataset.classes
print("类别标签：", class_names)


model = ResNet18(len(class_names))

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, val_loader, criterion, optimizer, epoch_num, save_interval):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_f1 = 0.0

    for epoch in range(epoch_num):
        print(f"Epoch {epoch + 1}/{epoch_num}")
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0
            all_preds = []
            all_labels = []

            for inputs, labels in tqdm(dataloader, desc=f"{phase} Epoch {epoch + 1}", leave=False):
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.data.cpu().numpy())

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)
            epoch_f1 = f1_score(all_labels, all_preds, average='weighted')  # 计算 F1 分数

            logging.info(f"{phase} Epoch {epoch + 1}/{epoch_num}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}, F1 Score: {epoch_f1:.4f}")

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} F1 Score: {epoch_f1:.4f}")

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_f1 = epoch_f1
                best_model_wts = copy.deepcopy(model.state_dict())

                best_model_path = os.path.join('model', f'best_model_epoch_{epoch + 1}_best.pth')
                torch.save(model.state_dict(), best_model_path)
                print(f"最佳模型已保存到 {best_model_path}")

        if (epoch + 1) % save_interval == 0:
            save_path = os.path.join('model', f'model_epoch_{epoch + 1}.pth')
            torch.save(model.state_dict(), save_path)
            print(f"模型已保存到 {save_path}")

        print()

    print('Best val Acc: {:4f}, Best val F1 Score: {:4f}'.format(best_acc, best_f1))

    model.load_state_dict(best_model_wts)
    return model

trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, epoch_num, save_interval)

final_model_path = os.path.join('model', f'final_model_epoch_{epoch_num}.pth')
torch.save(trained_model.state_dict(), final_model_path)
logging.info(f"最终模型已保存到 {final_model_path}")
