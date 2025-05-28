import os
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import torch
from torchvision import transforms
from resnet18 import ResNet18
import pytorch_grad_cam 
from pytorch_grad_cam.utils.image import show_cam_on_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['cat', 'dog', 'other']
model = ResNet18(len(class_names))
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, len(class_names))
model.load_state_dict(torch.load('./model/best_model.pth', weights_only=True))
model = model.to(device)
model.eval()
target_layers = [model.layer4[0].bn2] # 0

trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(224),
    transforms.CenterCrop(224)
])
normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

data_root = './Animal_recognition/val'
save_root = './feature_maps'
categories = os.listdir(data_root)

cam = pytorch_grad_cam.GradCAMPlusPlus(model=model, target_layers=target_layers)

for category in categories:
    save_dir = os.path.join(save_root, 'feature_maps', category)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    img_folder = os.path.join(data_root, category)
    for img_name in os.listdir(img_folder):
        img_path = os.path.join(img_folder, img_name)
        
        try:
            origin_img = cv2.imread(img_path)
            if origin_img is None:
                continue
            
            rgb_img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2RGB)
            
            crop_img = trans(rgb_img).to(device)
            net_input = normalize(crop_img).unsqueeze(0)
            
            canvas_img = (crop_img * 255).byte().cpu().numpy().transpose(1, 2, 0)
            canvas_img = cv2.cvtColor(canvas_img, cv2.COLOR_RGB2BGR)
            
            grayscale_cam = cam(net_input)[0, :]
            src_img = np.float32(canvas_img) / 255
            
            visualization_img = show_cam_on_image(src_img, grayscale_cam, use_rgb=False)
            
            save_path = os.path.join(save_dir, img_name)
            cv2.imwrite(save_path, visualization_img)
            
        except Exception as e:
            print(f"处理图片{img_name}时出错: {str(e)}")

