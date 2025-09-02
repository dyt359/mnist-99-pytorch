# mnist-99-pytorch

> 博客：[https://blog.csdn.net/2301_79914577/article/details/151108781?fromshare=blogdetail&sharetype=blogdetail&sharerId=151108781&sharerefer=PC&sharesource=2301_79914577&sharefrom=from_link]

## 1. 一键跑通
```bash
git clone https://github.com/<your_name>/mnist-99-pytorch.git
cd mnist-99-pytorch
conda env create -f env.yml
conda activate mnist_gpu
#如果有嫌训练100次的时间太长,可以改完30次,30次左右之后学习率就下降了
python mnist_gpu.py          # 训练 100 epoch，得到 model_weights.pth
python Download_test_set.py  # 导出 10k PNG 到 mnist_test_png/
python infer_folder.py       # 拖拽式推理 GUI
