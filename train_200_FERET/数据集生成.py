import os
import shutil
import random


def split_feret_data(data_dir, fa_dir, fb_dir):
    if not os.path.exists(fa_dir):
        os.makedirs(fa_dir)
    if not os.path.exists(fb_dir):
        os.makedirs(fb_dir)

    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if os.path.isdir(folder_path):
            images = [f for f in os.listdir(folder_path) if f.endswith('.tif')]
            if len(images) >= 2:
                fa_image = random.choice(images)
                fb_image = random.choice([img for img in images if img != fa_image])

                fa_image_path = os.path.join(folder_path, fa_image)
                fb_image_path = os.path.join(folder_path, fb_image)

                fa_save_path = os.path.join(fa_dir, f'{folder}_fa.tif')
                fb_save_path = os.path.join(fb_dir, f'{folder}_fb.tif')

                shutil.copy(fa_image_path, fa_save_path)
                shutil.copy(fb_image_path, fb_save_path)

                # 打印信息以调试
                print(f'Copied {fa_image_path} to {fa_save_path}')
                print(f'Copied {fb_image_path} to {fb_save_path}')


# 定义路径
data_dir = r'C:\Users\33455\Desktop\附加题\feret\train_200_FERET\train'
fa_dir = r'C:\Users\33455\Desktop\附加题\feret\fa'
fb_dir = r'C:\Users\33455\Desktop\附加题\feret\fb'

# 执行拆分
split_feret_data(data_dir, fa_dir, fb_dir)