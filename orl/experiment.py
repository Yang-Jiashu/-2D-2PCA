import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, BallTree
from sklearn.metrics import accuracy_score
from load_data import load_orl_data
from preprocess_data import preprocess_images
from two_d2pca import compute_2d2pca, project_2d2pca

data_dir = r'C:\Users\33455\Desktop\附加题\orl\archive'
X, y = load_orl_data(data_dir)
X = preprocess_images(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify=y, random_state=42)

# 计算(2D)²PCA的特征向量
row_eigvecs, col_eigvecs = compute_2d2pca(X_train)

# 将图像投影到(2D)²PCA特征向量空间
projected_train_images = project_2d2pca(X_train, row_eigvecs, col_eigvecs)
projected_test_images = project_2d2pca(X_test, row_eigvecs, col_eigvecs)

# 将投影后的图像展平
flattened_train_images = np.array([c.flatten() for c in projected_train_images])
flattened_test_images = np.array([c.flatten() for c in projected_test_images])

# 使用BallTree进行分类
tree = BallTree(flattened_train_images)
dist, ind = tree.query(flattened_test_images, k=1)

# 计算准确率
y_pred = y_train[ind[:, 0]]
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')