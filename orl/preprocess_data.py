import numpy as np

def preprocess_data(images):
    mean_image = np.mean(images, axis=0)
    images_centered = images - mean_image
    return images_centered, mean_image

if __name__ == "__main__":
    from load_data import load_data

    data_dir = r'C:\Users\33455\Desktop\附加题\orl\archive'
    (train_images, train_labels), (test_images, test_labels) = load_data(data_dir)

    train_images_centered, mean_image = preprocess_data(train_images)
    test_images_centered = test_images - mean_image

    print(f'Mean image shape: {mean_image.shape}')
    print(f'Centered training data shape: {train_images_centered.shape}')
    print(f'Centered test data shape: {test_images_centered.shape}')
