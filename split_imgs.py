# 白底黑字变黑底白字
import os
import shutil
from pathlib import Path

# ========== 参数配置 ==========
SRC_DIR   = r'./mnist_test_png'      # 原始图片所在目录
DST_ROOT  = r'./mnist_test_png' # 拆分后 10 个子文件夹的父目录
BATCH     = 1000                    # 每个子文件夹的图片数量
# ==============================


def split_images():
    src_path = Path(SRC_DIR)
    dst_root = Path(DST_ROOT)

    # 找到所有图片（可按后缀自行增删）
    img_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif'}
    imgs = [f for f in src_path.iterdir() if f.is_file() and f.suffix.lower() in img_ext]
    imgs.sort(key=lambda p: p.name)  # 按文件名排序

    total = len(imgs)
    if total == 0:
        print('未找到任何图片文件，请检查源目录及后缀匹配规则。')
        return

    print(f'共检测到 {total} 张图片，开始拆分...')

    # 确保输出根目录存在
    dst_root.mkdir(parents=True, exist_ok=True)

    for idx, img_path in enumerate(imgs):
        batch_no = idx // BATCH + 1          # 第几批（1~10）
        sub_dir  = dst_root / f'batch_{batch_no:02d}'
        sub_dir.mkdir(exist_ok=True)

        dst_path = sub_dir / img_path.name
        shutil.move(str(img_path), str(dst_path))   # 如需复制，改成 shutil.copy

    print('全部完成！')


if __name__ == '__main__':
    split_images()