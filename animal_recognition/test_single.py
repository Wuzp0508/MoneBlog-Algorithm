import torch
from torchvision import transforms, models
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_names = ['cat', 'dog', 'other']

model = ResNet18(len(class_names))
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, len(class_names))
model.load_state_dict(torch.load('./model/best_model.pth'))
model = model.to(device)
model.eval()

def predict(model, img_path):
    transform_predict = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    img = Image.open(img_path)
    img = transform_predict(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        _, preds = torch.max(outputs, 1)
    
    return class_names[preds.item()]

img_path = './Animal_recognition/val/cat/cat.10019.jpg'
predicted_class = predict(model, img_path)
print(f"预测的类别是: {predicted_class}")
