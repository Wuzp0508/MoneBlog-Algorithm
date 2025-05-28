import torch
from torchvision import transforms
from resnet18 import ResNet18
from PIL import Image
import os
from sklearn.metrics import f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_names = ['cat', 'dog', 'other']

model = ResNet18(len(class_names))
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, len(class_names))
model.load_state_dict(torch.load('./model/best_model.pth', weights_only=True))
model = model.to(device)
model.eval()

def predict(model, img_path):
    transform_predict = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    try:
        img = Image.open(img_path).convert("RGB")
        img = transform_predict(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img)
            _, preds = torch.max(outputs, 1)

        return class_names[preds.item()]
    except Exception as e:
        print(f"无法预测图片 {img_path}: {e}")
        return None

def predict_folder(model, folder_path):
    predictions = {}
    supported_formats = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")  # 支持的图片格式

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(supported_formats):
                img_path = os.path.join(root, file)
                pred = predict(model, img_path)
                if pred is not None:
                    predictions[img_path] = pred
    return predictions

def calculate_metrics(predictions, true_labels):
    class_preds = []
    class_labels = []

    for img_path, pred_class in predictions.items():
        true_class = os.path.basename(os.path.dirname(img_path))
        class_preds.append(pred_class)
        class_labels.append(true_class)

    accuracy = sum(p == t for p, t in zip(class_preds, class_labels)) / len(class_labels) if class_labels else 0
    f1 = f1_score(class_labels, class_preds, average='weighted', labels=class_names)

    return accuracy, f1, class_labels, class_preds

def evaluate_model_on_test_set(model, test_folder):
    total_correct = 0
    total_images = 0
    all_preds = []
    all_labels = []

    for class_folder in os.listdir(test_folder):
        class_folder_path = os.path.join(test_folder, class_folder)
        if os.path.isdir(class_folder_path):  # 只处理文件夹
            print(f"正在预测类别 '{class_folder}' 中的图片...")
            predictions = predict_folder(model, class_folder_path)

            accuracy, f1, class_labels, class_preds = calculate_metrics(predictions, class_folder)
            print(f"类别 '{class_folder}' 的准确率: {accuracy * 100:.2f}%")
            print(f"类别 '{class_folder}' 的F1分数: {f1:.4f}")
            
            all_labels.extend(class_labels)
            all_preds.extend(class_preds)
            
            correct = 0
            for img_path, pred_class in predictions.items():
                if os.path.basename(os.path.dirname(img_path)) == pred_class:
                    correct += 1

            total_images += len(predictions)
            total_correct += correct
            
            print(f"类别总数: {len(predictions)}")
            print(f"类别正确数: {correct}")
            
    print(f"全体总数: {total_images}")
    print(f"类别正确数: {total_correct}")
    overall_accuracy = total_correct / total_images if total_images > 0 else 0
    overall_f1 = f1_score(all_labels, all_preds, average='weighted', labels=class_names)
    print(f"整体准确率: {overall_accuracy * 100:.2f}%")
    print(f"整体F1分数: {overall_f1:.4f}")

test_folder_path = './Animal_recognition/val'
evaluate_model_on_test_set(model, test_folder_path)
