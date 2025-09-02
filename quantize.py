import torch
from mnist_gpu import Net

model = Net()
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

quantized = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
torch.save(quantized.state_dict(), 'model_weights_int8.pth')
print('量化完成，大小约 ↓75 %，已保存 model_weights_int8.pth')