import numpy as np
from two_dpca import compute_2dpca

def select_eigenvectors(eigvals, eigvecs, theta=0.95):
    # 计算特征值的累计和
    total_variance = np.sum(eigvals)
    variance_ratio = eigvals / total_variance
    cumulative_variance_ratio = np.cumsum(variance_ratio)

    # 找到累计方差贡献率达到theta的特征向量数量
    num_vectors = np.searchsorted(cumulative_variance_ratio, theta) + 1
    return eigvecs[:, :num_vectors]

def compute_2d2pca(X, theta=0.95):
    # 计算行方向和列方向的特征向量
    row_eigvals, row_eigvecs = compute_2dpca(X)
    col_eigvals, col_eigvecs = compute_2dpca(np.array([x.T for x in X]))

    # 根据累计方差贡献率选择特征向量
    row_eigvecs = select_eigenvectors(row_eigvals, row_eigvecs, theta)
    col_eigvecs = select_eigenvectors(col_eigvals, col_eigvecs, theta)
    return row_eigvecs, col_eigvecs

def project_2d2pca(X, row_eigvecs, col_eigvecs):
    return [col_eigvecs.T @ x @ row_eigvecs for x in X]

if __name__ == "__main__":
    from load_data import load_orl_data
    from preprocess_data import preprocess_images

    data_dir = r'C:\Users\33455\Desktop\附加题\archive'
    X, y = load_orl_data(data_dir)
    X = preprocess_images(X)

    row_eigvecs, col_eigvecs = compute_2d2pca(X, theta=0.95)
    projected_X = project_2d2pca(X, row_eigvecs, col_eigvecs)
    print(f'(2D)²PCA computed and projected. Projected X shape: {np.array(projected_X).shape}')
