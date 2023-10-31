import dataset
import numpy as np
import matplotlib.pyplot as plt
import model
import torch
import os
from PIL import Image
from torchvision import transforms
#�����ڲ��Լ��ϣ�����ͼƬ��Ԥ����Ҫ����һ��py�ļ�
# ����transforms ����ģ�� ��ȡͼ����transforms���� ��model��ͼ����д��� ��ѡȡ����ܵģ������

test_transform=transforms.Compose([
    transforms.Resize((227,227)),  #Ҫ��,
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
