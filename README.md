# 2D-2PCA

这个项目实现了2D-2PCA算法。

## 目录

- [简介](#简介)
- [安装](#安装)
- [使用](#使用)
- [贡献](#贡献)
- [许可证](#许可证)

## 简介

此项目是对2D-2PCA算法的实现，用于降维和特征提取。具体实现包括数据加载、预处理、2DPCA 和 (2D)^2PCA 算法，以及实验代码。

### `load_data.py`

该模块负责从 FERET 数据集加载数据。它遍历数据目录中的图像文件，将它们转换为灰度图像并调整大小，然后将图像数据和标签分别存储为训练集和测试集。

#### 示例代码：
```python
from load_data import load_feret_data

data_dir = r'C:\Users\33455\Desktop\附加题\feret\dataset'
(train_images, train_labels), (test_images, test_labels) = load_feret_data(data_dir)

print(f'Training data shape: {train_images.shape}')
print(f'Test data shape: {test_images.shape}')
