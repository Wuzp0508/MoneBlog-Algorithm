import os 
from coze import Coze
import re
import cv2
import numpy as np
import requests
import sys
import io

def resize_img(image_path):
    image_response = requests.get(image_path)
    if image_response.status_code == 200:
        image_data = np.frombuffer(image_response.content, dtype=np.uint8)
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    image_height, image_width = image.shape[:2]

    new_width, new_height = 1920, 1080
    resize_image = cv2.resize(image, (new_width, new_height))
    resize_image_height, resize_image_width = resize_image.shape[:2]
    return resize_image

def LLM_animal_recognition(image_path):
    os.environ['COZE_API_TOKEN'] = 'pat_3FicWyWyjYsgaFtAr26b72NYGlRjexnPWW4FQ9RVi8lRwtUerZkRf407bqYmEsHB'
    os.environ['COZE_BOT_ID'] = "7501131642988183604"
    response = None
    old_stdout = sys.stdout
    try:
        sys.stdout = io.StringIO()
        chat = Coze(
            api_token=os.environ['COZE_API_TOKEN'],
            bot_id=os.environ['COZE_BOT_ID'],
            max_chat_rounds=20,
            stream=True
        )
        response = chat(image_path)
        if "[" in response:
            result = re.search(r"\[(.*?)\]", response).group(1).replace(" ", "").split(",")
            box_color = (255, 0, 255)
            resized_img = resize_img(image_path)
            
            try:
                x1, y1, x2, y2 = [int(coord) for coord in result]
            except ValueError as e:
                return f"Error: Invalid coordinates in LLM response - {result}"
            
            resized_img = resize_img(image_path)
            cv2.rectangle(resized_img, (x1, y1), (x2, y2), color=box_color, thickness=2)     
                   
            image_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
            if not os.path.exists(image_dir):
                os.makedirs(image_dir)
            output_file = os.path.join(image_dir, "processed_image.png")
            cv2.imwrite(output_file, resized_img)
            return [response, resized_img]
    finally:
        # 恢复标准输出
        sys.stdout = old_stdout
