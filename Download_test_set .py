# 下载测试集
import os
from torchvision import datasets, transforms
from PIL import Image

out_dir = 'mnist_test_png'
os.makedirs(out_dir, exist_ok=True)

# 下载/加载测试集（如果已下载会直接用本地缓存）
test_set = datasets.MNIST(root='./data', train=False, download=True,
                          transform=transforms.ToTensor())

for idx, (img_tensor, label) in enumerate(test_set):
    # tensor -> PIL -> PNG
    img = transforms.ToPILImage()(img_tensor)
    save_path = os.path.join(out_dir, f'{label}_{idx:05d}.png')
    img.save(save_path)

print(f'已导出 {len(test_set)} 张图片到 {out_dir}')