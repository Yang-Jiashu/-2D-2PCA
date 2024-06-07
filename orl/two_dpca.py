import numpy as np

def compute_2dpca(X):
    # 计算图像均值
    mean_image = np.mean(X, axis=0)
    # 计算协方差矩阵
    cov_matrix = np.zeros((mean_image.shape[1], mean_image.shape[1]))
    for img in X:
        diff = img - mean_image
        cov_matrix += np.dot(diff.T, diff)
    cov_matrix /= X.shape[0]
    # 对协方差矩阵进行特征值分解
    eigvals, eigvecs = np.linalg.eigh(cov_matrix)
    # 返回特征值和特征向量（即投影矩阵）
    return eigvals, eigvecs

def project_2dpca(X, eigvecs):
    # 投影图像到特征向量空间
    return [x @ eigvecs for x in X]

if __name__ == "__main__":
    from load_data import load_orl_data
    from preprocess_data import preprocess_images

    data_dir = 'C:\\Users\\33455\\Desktop\\附加题\\archive'
    X, y = load_orl_data(data_dir)
    X = preprocess_images(X)

    eigvals, eigvecs = compute_2dpca(X)
    projected_X = project_2dpca(X, eigvecs)
    print(f'2DPCA computed and projected. Projected X shape: {np.array(projected_X).shape}')