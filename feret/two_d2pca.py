import numpy as np


class TwoD2PCA:
    def __init__(self, row_components, col_components):
        self.row_components = row_components
        self.col_components = col_components
        self.row_proj_matrix = None
        self.col_proj_matrix = None

    def fit(self, images):
        row_cov_matrix = np.zeros((60, 60))
        col_cov_matrix = np.zeros((60, 60))

        for image in images:
            image = image.reshape(60, 60)
            row_cov_matrix += np.dot(image.T, image)
            col_cov_matrix += np.dot(image, image.T)

        row_cov_matrix /= images.shape[0]
        col_cov_matrix /= images.shape[0]

        row_eig_values, row_eig_vectors = np.linalg.eigh(row_cov_matrix)
        col_eig_values, col_eig_vectors = np.linalg.eigh(col_cov_matrix)

        row_sorted_indices = np.argsort(row_eig_values)[::-1]
        col_sorted_indices = np.argsort(col_eig_values)[::-1]

        self.row_proj_matrix = row_eig_vectors[:, row_sorted_indices[:self.row_components]]
        self.col_proj_matrix = col_eig_vectors[:, col_sorted_indices[:self.col_components]]

    def transform(self, images):
        projected_images = []
        for image in images:
            image = image.reshape(60, 60)
            projected_image = np.dot(np.dot(self.col_proj_matrix.T, image), self.row_proj_matrix)
            projected_images.append(projected_image.flatten())
        return np.array(projected_images)


if __name__ == "__main__":
    from load_data import load_feret_data
    from preprocess_data import preprocess_data

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

    print(f'Projected training data shape: {train_projected.shape}')
    print(f'Projected test data shape: {test_projected.shape}')