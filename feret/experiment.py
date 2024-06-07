import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from load_data import load_feret_data
from preprocess_data import preprocess_data
from two_d2pca import TwoD2PCA


def run_experiment():
    data_dir = 'C:\\Users\\33455\\Desktop\\附加题\\feret'
    (train_images, train_labels), (test_images, test_labels) = load_feret_data(data_dir)

    train_images_centered, mean_image = preprocess_data(train_images)
    test_images_centered = test_images - mean_image

    train_images_centered = train_images_centered.reshape(-1, 60, 60)
    test_images_centered = test_images_centered.reshape(-1, 60, 60)

    two_d2pca = TwoD2PCA(row_components=13, col_components=14)
    two_d2pca.fit(train_images_centered)
    train_projected = two_d2pca.transform(train_images_centered).reshape(train_images_centered.shape[0], -1)
    test_projected = two_d2pca.transform(test_images_centered).reshape(test_images_centered.shape[0], -1)

    print(f'Train labels: {train_labels}')
    print(f'Test labels: {test_labels}')

    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(train_projected, train_labels)

    accuracy = knn.score(test_projected, test_labels)
    print(f'Accuracy: {accuracy * 100:.2f}%')


if __name__ == "__main__":
    run_experiment()