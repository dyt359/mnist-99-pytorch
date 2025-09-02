import os
import sys
import time
import threading
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageTk
import numpy as np

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    from tkinterdnd2 import DND_FILES, TkinterDnD
except ImportError:
    print("请先 pip install tkinterdnd2")
    sys.exit()

# ---------------- 设备 ----------------
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = 'model_weights.pth'          # 训练好的 state_dict

# ---------------- 网络（与训练时一致） ----------------
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# --------------- 加载模型 ---------------
model = Net().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# --------------- 预处理 ---------------
transform = T.Compose([
    T.Grayscale(num_output_channels=1),
    T.Resize((28, 28)),
    T.ToTensor(),
    T.Normalize((0.5,), (0.5,))
])

# --------------- 工具函数 ---------------
def auto_invert_if_needed(img_pil):
    """保证黑底白字"""
    img_np = np.array(img_pil)
    if img_np.mean() > 127:
        img_np = 255 - img_np
    return Image.fromarray(img_np, mode='L')

def list_images(path):
    """返回目录或单文件下所有图片绝对路径"""
    path = Path(path)
    if path.is_file():
        return [str(path)] if path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'] else []
    return [str(p) for p in path.rglob('*') if p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']]

# --------------- 主窗口 ---------------
class MainApp(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()
        self.title("MNIST 批量识别")
        self.geometry("500x400")
        self.resizable(False, False)

        self.paths = []          # 待识别文件列表
        self.results = []        # (path, pred, conf)
        self.max_batch = 2000    # ★ 内存软上限，可自行调整

        # ---- 顶部控制区 ----
        frm = ttk.Frame(self)
        frm.pack(padx=10, pady=10, fill='x')
        ttk.Button(frm, text="选择图片…", command=self.select_files).pack(side='left', padx=5)
        ttk.Button(frm, text="选择文件夹…", command=self.select_folder).pack(side='left', padx=5)
        ttk.Button(frm, text="开始识别", command=self.run_recognition).pack(side='left', padx=5)
        ttk.Button(frm, text="一键清除", command=self.clear_all).pack(side='left', padx=5)

        # ---- 拖拽区域 ----
        drop_frame = ttk.LabelFrame(self, text="拖拽区域（图片或文件夹）", height=80)
        drop_frame.pack(fill='both', padx=10, pady=5)
        drop_frame.pack_propagate(False)
        self.drop_lbl = ttk.Label(drop_frame, anchor='center', foreground='gray')
        self.drop_lbl.pack(fill='both', expand=True)
        self.drop_lbl["text"] = "把图片或文件夹拖到这里"
        self.drop_lbl.drop_target_register(DND_FILES)
        self.drop_lbl.dnd_bind('<<Drop>>', self.on_drop)

        # ---- 结果显示表格 ----
        self.tree = ttk.Treeview(self, columns=('pred', 'conf'), show='tree headings', height=10)
        self.tree.heading('#0', text='文件名')
        self.tree.heading('pred', text='预测')
        self.tree.heading('conf', text='置信度')
        self.tree.column('#0', width=260)
        self.tree.column('pred', width=80, anchor='center')
        self.tree.column('conf', width=120, anchor='center')
        scroll = ttk.Scrollbar(self, orient='vertical', command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(fill='both', padx=10, pady=5, expand=True)
        scroll.pack(side='right', fill='y')

        # ★ 双击弹窗
        self.tree.bind('<Double-1>', self.on_row_dbl_click)

    # ------------- 文件/文件夹选择 -------------
    def select_files(self):
        files = filedialog.askopenfilenames(
            title="选择图片",
            filetypes=[('图片', '*.png *.jpg *.jpeg *.bmp *.tif *.tiff')]
        )
        self.paths.extend(list(files))
        self.update_drop_label()

    def select_folder(self):
        folder = filedialog.askdirectory(title="选择文件夹")
        if folder:
            self.paths.extend(list_images(folder))
            self.update_drop_label()

    # ------------- 拖拽 -------------
    def on_drop(self, event):
        raw = self.tk.splitlist(event.data)
        for r in raw:
            r = r.strip('{}')
            if os.path.isdir(r):
                self.paths.extend(list_images(r))
            else:
                self.paths.extend(list_images(r))
        self.update_drop_label()

    def update_drop_label(self):
        self.paths = list(dict.fromkeys(self.paths))
        if len(self.paths) > self.max_batch:
            messagebox.showwarning("提示", f"一次最多处理 {self.max_batch} 张，已自动截断。")
            self.paths = self.paths[:self.max_batch]
        self.drop_lbl["text"] = f"已添加 {len(self.paths)} 个文件，点击“开始识别”"

    # ------------- 识别 -------------
    def run_recognition(self):
        if not self.paths:
            messagebox.showwarning("提示", "请先添加图片或文件夹！")
            return
        self.results.clear()
        ProgressWindow(self, self.paths)

    # ------------- 展示结果（低置信度排前） -------------
    def fill_results(self, results):
        # ★ 按置信度升序排序（低的在前）
        results.sort(key=lambda x: x[2])  # x[2] 是 conf
        self.results = results
        for item in self.tree.get_children():
            self.tree.delete(item)
        for path, pred, conf in results:
            name = Path(path).name
            self.tree.insert('', 'end', text=name, values=(pred, f"{conf:.2%}"))
        messagebox.showinfo("完成", f"共识别 {len(results)} 张图片，已按置信度升序排列")

    # ------------- 一键清除 -------------
    def clear_all(self):
        self.paths.clear()
        self.results.clear()
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.drop_lbl["text"] = "把图片或文件夹拖到这里"

    # ------------- 双击弹窗 -------------
    def on_row_dbl_click(self, event):
        selected = self.tree.selection()
        if not selected:
            return
        idx = self.tree.index(selected[0])
        if 0 <= idx < len(self.results):
            path, pred, conf = self.results[idx]
            DetailWindow(self, path, pred, conf)

# --------------- 进度条子窗口 ---------------
class ProgressWindow(tk.Toplevel):
    def __init__(self, master, paths):
        super().__init__(master)
        self.title("识别中……")
        self.geometry("300x120")
        self.resizable(False, False)
        self.grab_set()

        ttk.Label(self, text="正在推理，请稍候……").pack(pady=10)
        self.bar = ttk.Progressbar(self, length=250, mode='determinate')
        self.bar.pack(pady=5)
        self.count_lbl = ttk.Label(self, text="0 / 0")
        self.count_lbl.pack()

        self.paths = paths
        self.results = []

        thread = threading.Thread(target=self.work, daemon=True)
        thread.start()
        self.after(100, self.check_thread, thread)

    def work(self):
        total = len(self.paths)
        for idx, p in enumerate(self.paths, 1):
            try:
                img_pil = Image.open(p).convert('L')
                img_pil = auto_invert_if_needed(img_pil)
                tensor = transform(img_pil).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    logits = model(tensor)
                    prob = torch.softmax(logits, dim=1)
                    pred = prob.argmax(dim=1).item()
                    conf = prob[0, pred].item()
                self.results.append((p, pred, conf))
            except Exception as e:
                print(f"跳过 {p}: {e}")
            self.bar["value"] = idx / total * 100
            self.count_lbl["text"] = f"{idx} / {total}"
            time.sleep(0.01)

    def check_thread(self, t):
        if t.is_alive():
            self.after(100, self.check_thread, t)
        else:
            self.destroy()
            self.master.fill_results(self.results)

# --------------- 详情子窗口 ---------------
class DetailWindow(tk.Toplevel):
    def __init__(self, master, img_path, pred, conf):
        super().__init__(master)
        self.title(Path(img_path).name)
        self.geometry("350x200")
        self.resizable(False, False)

        img = Image.open(img_path).convert('L')
        img = auto_invert_if_needed(img).resize((150, 150), Image.NEAREST)
        img_tk = ImageTk.PhotoImage(img)

        left = ttk.Label(self, image=img_tk)
        left.image = img_tk  # 防止被垃圾回收
        left.pack(side='left', padx=10, pady=10)

        right = ttk.Frame(self)
        right.pack(side='left', padx=10, pady=10, fill='y')

        ttk.Label(right, text="预测数字", font=('Arial', 14)).pack(pady=5)
        ttk.Label(right, text=str(pred), font=('Arial', 36, 'bold')).pack()
        ttk.Label(right, text="置信率", font=('Arial', 14)).pack(pady=(10, 5))
        ttk.Label(right, text=f"{conf:.2%}", font=('Arial', 16)).pack()

# --------------- 入口 ---------------
if __name__ == '__main__':
    app = MainApp()
    app.mainloop()