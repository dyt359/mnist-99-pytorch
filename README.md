# mnist-99-pytorch

> 博客：[稍后替换为你的地址]

## 1. 一键跑通
```bash
git clone https://github.com/<your_name>/mnist-99-pytorch.git
cd mnist-99-pytorch
conda env create -f env.yml
conda activate mnist_gpu
python mnist_gpu.py          # 训练 100 epoch，得到 model_weights.pth
python Download_test_set.py  # 导出 10k PNG 到 mnist_test_png/
python infer_folder.py       # 拖拽式推理 GUI