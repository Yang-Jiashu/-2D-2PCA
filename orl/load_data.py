import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

def load_data(data_dir):
    images = []
    labels = []
    for person_dir in os.listdir(data_dir):
        person_path = os.path.join(data_dir, person_dir)
        if os.path.isdir(person_path):
            for image_name in os.listdir(person_path):
                if image_name.endswith('.pgm'):
                    image_path = os.path.join(person_path, image_name)
                    image = Image.open(image_path).convert('L')
                    image = image.resize((60, 60))  # 调整图像大小
                    images.append(np.array(image).flatten())
                    labels.append(person_dir)

    images = np.array(images)
    labels = np.array(labels)

    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size=0.5, stratify=labels, random_state=42
    )

    return (train_images, train_labels), (test_images, test_labels)

if __name__ == "__main__":
    data_dir = r'C:\Users\33455\Desktop\附加题\orl\archive'
    (train_images, train_labels), (test_images, test_labels) = load_data(data_dir)
    print(f'Training data shape: {train_images.shape}')
    print(f'Test data shape: {test_images.shape}')
