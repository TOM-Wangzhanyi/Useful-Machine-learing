import dataset
import numpy as np
import matplotlib.pyplot as plt
import model
import torch
import os
from PIL import Image
from torchvision import transforms
#区别于测试集合，单个图片的预测需要单独一个py文件
# 定义transforms 加载模型 获取图像并用transforms处理 用model对图像进行处理 再选取最可能的，输出。

test_transform=transforms.Compose([
    transforms.Resize((227,227)),  #要有,
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5,.5,.5], std=[.5, .5, .5])
])


my_path=os.abspath(os.path.join(os.getcwd(),"."))


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
net=model.AlexNet()
net.load_state_dict(torch.load(os.path.join(my_path,"alexnet.path")))
net.to(device)

test_path = os.path.join(my_path,"data/catVSdog/test_data/cat/cat.10000.jpg")
test_img=Image.open(test_path).convert("RGB")
test_img = test_transform(test_img).to(device)

result=net(test_img)
choice=["cat","dog"]
predict = choice[result.argmax().item()]
print(predict)
