import os
import numpy as np
from PIL import Image

# 定义图像尺寸
image_size = (112, 92)

def load_orl_data(data_dir):
    X_train, y_train, X_test, y_test = [], [], [], []
    label = 0
    for person_dir in os.listdir(data_dir):
        person_path = os.path.join(data_dir, person_dir)
        if os.path.isdir(person_path):
            images = []
            for image_name in sorted(os.listdir(person_path)):
                if image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
                    image_path = os.path.join(person_path, image_name)
                    try:
                        image = Image.open(image_path).convert('L')
                        image = image.resize(image_size)
                        images.append(np.array(image))
                    except Exception as e:
                        print(f"Error loading image {image_path}: {e}")
            if len(images) >= 10:  # 确保每个人至少有10张图像
                # 前五张图像用于训练，后五张图像用于测试
                X_train.extend(images[:5])
                y_train.extend([label] * 5)
                X_test.extend(images[5:])
                y_test.extend([label] * 5)
                label += 1
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    return X_train, y_train, X_test, y_test

if __name__ == "__main__":
    data_dir = r'C:\Users\33455\Desktop\附加题\orl\archive'
    X_train, y_train, X_test, y_test = load_orl_data(data_dir)
    print(f'Data loaded. Training set size: {X_train.shape}, Testing set size: {X_test.shape}')