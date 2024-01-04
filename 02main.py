import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from fastapi import FastAPI, File, UploadFile
from typing import List, Dict
import uvicorn

app = FastAPI()

# 画像の変換関数
def png_to_mnist_equivalent(png_file):
    img = Image.open(png_file)
    img_reverse = Image.fromarray(np.array(img) ^ 255)

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])

    mnist_equivalent_img = transform(img_reverse)
    return img, mnist_equivalent_img

# ニューラルネットワークモデルの定義と推論
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(3)
        self.fc = nn.Linear(588, 10)

    def forward(self, x):
        h = self.conv(x)
        h = F.relu(h)
        h = self.bn(h)
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        h = h.view(-1, 588)
        h = self.fc(h)
        return h

# POSTリクエストで'/predict'エンドポイントを作成し、画像の予測を行う
@app.post("/predict")
async def predict(files: List[UploadFile] = File(...)):
    results = []
    for file in files:
        original_img, mnist_equivalent_image = png_to_mnist_equivalent(file.file)

        net = Net().cpu().eval()
        net.load_state_dict(torch.load('mnist.pt', map_location=torch.device('cpu')))
        mnist_equivalent_image = mnist_equivalent_image.unsqueeze(0)
        y = net(mnist_equivalent_image)
        y = F.softmax(y, dim=1)
        prediction = torch.argmax(y)

        results.append({
            "original_image_path": file.filename,
            "prediction": prediction.item()
        })

    return results

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)