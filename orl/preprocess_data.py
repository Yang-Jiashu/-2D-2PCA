import numpy as np

def preprocess_images(X):
    if len(X) == 0:
        return X
    # 将图像数据标准化为0到1之间
    X = X.astype('float32') / 255.0
    # 平均值归零
    X -= np.mean(X, axis=0)
    # 标准差归一
    X /= np.std(X, axis=0)
    return X

if __name__ == "__main__":
    from load_data import load_orl_data

    data_dir = r'C:\Users\33455\Desktop\附加题\orl\archive'
    X_train, y_train, X_test, y_test = load_orl_data(data_dir)
    print(f'Data loaded. Training set size: {X_train.shape}, Testing set size: {X_test.shape}')
    X_train = preprocess_images(X_train)
    X_test = preprocess_images(X_test)
    print(f'Data preprocessed. Training set shape: {X_train.shape}, Testing set shape: {X_test.shape}')