import torch
import torchvision
from tqdm import tqdm
import matplotlib.pyplot as plt

# ---------------- 网络加深 + 加宽 + Dropout ----------------
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.features = torch.nn.Sequential(
            # 28×28
            torch.nn.Conv2d(1, 32, 3, 1, 1),   # 32 通道
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),             # 14×14

            torch.nn.Conv2d(32, 64, 3, 1, 1),  # 64 通道
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),             # 7×7

            torch.nn.Conv2d(64, 128, 3, 1, 1), # 128 通道
            torch.nn.ReLU(),
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(128 * 7 * 7, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),             # Dropout 50 %
            torch.nn.Linear(256, 10)
            # 去掉 Softmax，CrossEntropyLoss 内部已做
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ---------------- 设备 ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- 数据增强 + 归一化 ----------------
train_transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomRotation(10),              # ±10° 旋转
    torchvision.transforms.RandomAffine(0, translate=(0.1, 0.1)),  # 平移 10 %
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,), (0.5,))
])

test_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,), (0.5,))
])

# ---------------- 数据集 & 加载器 ----------------
BATCH_SIZE = 256
EPOCHS = 100

train_data = torchvision.datasets.MNIST('./data', train=True,  download=True, transform=train_transform)
test_data  = torchvision.datasets.MNIST('./data', train=False, download=True, transform=test_transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_data,  batch_size=BATCH_SIZE)

# ---------------- 模型、损失、优化器 ----------------
net = Net().to(device)
print(net)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)   # 学习率调小

# ---------------- 训练循环 ----------------
history = {'test_loss': [], 'test_acc': []}
for epoch in range(1, EPOCHS + 1):
    net.train()
    running_loss, running_acc = 0.0, 0.0
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{EPOCHS}')

    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = net(imgs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        acc = (outputs.argmax(1) == labels).float().mean()
        running_loss += loss.item()
        running_acc  += acc.item()
        pbar.set_postfix(loss=running_loss / len(pbar), acc=running_acc / len(pbar))

    # ---------- 每个 epoch 结束后评估 ----------
    net.eval()
    correct, total_loss = 0, 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = net(imgs)
            total_loss += loss_fn(outputs, labels).item()
            correct += (outputs.argmax(1) == labels).sum().item()

    test_acc = correct / len(test_loader.dataset)
    test_loss = total_loss / len(test_loader)
    history['test_acc'].append(test_acc)
    history['test_loss'].append(test_loss)

    pbar.write(f'Epoch {epoch:02d}  Test Loss: {test_loss:.4f}  Test Acc: {test_acc:.4f}')

# ---------------- 画图 ----------------
plt.plot(history['test_loss'], label='Test Loss')
plt.legend(); plt.grid(); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.show()

plt.plot(history['test_acc'], color='red', label='Test Accuracy')
plt.legend(); plt.grid(); plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.show()

# ---------------- 保存 ----------------
# torch.save(net, './model_v2.pth')
torch.save(net.state_dict(), './model_weights.pth')