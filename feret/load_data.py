import os
import numpy as np
from PIL import Image

def load_feret_data(data_dir):
    train_images = []
    test_images = []
    train_labels = []
    test_labels = []

    for root, dirs, files in os.walk(data_dir):
        if 'fa' in root:
            for file in files:
                if file.endswith('.tif'):
                    image = Image.open(os.path.join(root, file)).convert('L')
                    image = image.resize((60, 60))
                    train_images.append(np.array(image).flatten())
                    label = os.path.basename(file).split('_')[0]
                    train_labels.append(label)
        elif 'fb' in root:
            for file in files:
                if file.endswith('.tif'):
                    image = Image.open(os.path.join(root, file)).convert('L')
                    image = image.resize((60, 60))
                    test_images.append(np.array(image).flatten())
                    label = os.path.basename(file).split('_')[0]
                    test_labels.append(label)

    return (np.array(train_images), np.array(train_labels)), (np.array(test_images), np.array(test_labels))

if __name__ == "__main__":
    data_dir = r'C:\Users\33455\Desktop\附加题\feret\dataset'
    (train_images, train_labels), (test_images, test_labels) = load_feret_data(data_dir)
    print(f'Training data shape: {train_images.shape}')
    print(f'Test data shape: {test_images.shape}')
