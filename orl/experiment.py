import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from load_data import load_data
from preprocess_data import preprocess_data
from two_dpca import TwoDPCA
from two_d2pca import TwoD2PCA

def run_experiment():
    data_dir = r'C:\Users\33455\Desktop\附加题\orl\archive'
    (train_images, train_labels), (test_images, test_labels) = load_data(data_dir)

    train_images_centered, mean_image = preprocess_data(train_images)
    test_images_centered = test_images - mean_image

    train_images_centered = train_images_centered.reshape(-1, 60, 60)
    test_images_centered = test_images_centered.reshape(-1, 60, 60)

    # 2DPCA
    two_dpca = TwoDPCA(threshold=0.95)
    two_dpca.fit(train_images_centered)
    train_projected_2dpca = two_dpca.transform(train_images_centered).reshape(train_images_centered.shape[0], -1)
    test_projected_2dpca = two_dpca.transform(test_images_centered).reshape(test_images_centered.shape[0], -1)

    # (2D)^2PCA
    two_d2pca = TwoD2PCA(threshold=0.95)
    two_d2pca.fit(train_images_centered)
    train_projected_2d2pca = two_d2pca.transform(train_images_centered).reshape(train_images_centered.shape[0], -1)
    test_projected_2d2pca = two_d2pca.transform(test_images_centered).reshape(test_images_centered.shape[0], -1)

    # 分类和准确性评估
    knn = KNeighborsClassifier(n_neighbors=1)

    # 2DPCA分类
    knn.fit(train_projected_2dpca, train_labels)
    accuracy_2dpca = knn.score(test_projected_2dpca, test_labels)
    print(f'2DPCA Accuracy: {accuracy_2dpca * 100:.2f}%')

    # (2D)^2PCA分类
    knn.fit(train_projected_2d2pca, train_labels)
    accuracy_2d2pca = knn.score(test_projected_2d2pca, test_labels)
    print(f'(2D)^2PCA Accuracy: {accuracy_2d2pca * 100:.2f}%')

if __name__ == "__main__":
    run_experiment()