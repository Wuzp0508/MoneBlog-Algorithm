import torch
from torchvision import transforms, models
from PIL import Image
import os
from .resnet18 import ResNet18
import requests
import numpy as np
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_names = ['cat', 'dog', 'other']

model = ResNet18(len(class_names))
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, len(class_names))

current_path = os.path.dirname(os.path.abspath(__file__))

model.load_state_dict(torch.load(os.path.join(current_path,'model/model.pth'), weights_only=True, map_location=device))
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
        img = None
        if(os.path.isfile(img_path)):
            img = Image.open(img_path).convert("RGB")
        elif(is_url(img_path)):
            img_response = requests.get(img_path)
            if img_response.status_code == 200:
                img_data = np.frombuffer(img_response.content, dtype=np.uint8)
                img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                # print(img.size)
        else:
            print(f"无法识别的图片路径: {img_path}")

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
    supported_formats = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")
    
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(supported_formats):
                img_path = os.path.join(root, file)
                pred = predict(model, img_path)
                if pred is not None:
                    predictions[img_path] = pred
    return predictions

def animal_recognition(image_path):
    # print(image_path)
    if(os.path.isfile(image_path)):
        print("输入的路径是一个文件")
        pred = predict(model, image_path)
        print(f"图片: {image_path} -> 预测类别: {pred}")
    elif os.path.isdir(image_path):
        print("输入的路径是一个文件夹")
        results = predict_folder(model, folder_path)
        for img_path, pred_class in results.items():
            print(f"图片: {img_path} -> 预测类别: {pred_class}")
    elif is_url(image_path):
        # print("输入的路径是一个URL")     
        pred = predict(model, image_path)
        # print(f"图片: {image_path} \n预测类别: {pred}")      
        return "预测类别: " + pred
    else:
        print("输入的路径既不是文件也不是文件夹, 也不是URL")

def is_url(text):
    try:
        from urllib.parse import urlparse
        result = urlparse(text)
        # 至少需要包含协议（如http/https）和域名/路径
        return all([result.scheme, result.netloc or result.path])
    except:
        return False
