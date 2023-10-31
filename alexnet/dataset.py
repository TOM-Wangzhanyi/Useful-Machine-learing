from torch.util.data import Dataset
from PIL import Image

class MyDataset(Dataset):  #�̳���Dataset�࣬Ҫ������������
    def __init__(self,text_path,transform=None,target_transform=None): # ����txt�ļ�����·���ͱ�ǩ�ֿ�
        img_path = open(text_path,"r")#Ȼ������ȫ�ֱ�������getitem����
        img=[]
        for line in img_path:
            line = line.strip().split(" ")
            img.append((line[0],int(line[1])))
        self.imgs = img   # self.imgsҲ���¶���ı�����Ŀ����Ϊ�˽�����ֲ���������ȥ����Ϊȫ�ֱ�������ĺ�������
        self.transform = transform
        self.target_transform=target_transform
    
    def __getitem__(self,index):# ������Щ·������img���м��أ��ٽ�img��label return��ȥ
        img_path , label = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        if(self.transform is not None):
            img = self.transform(img)
        return img , label
    
    def __len__(self):
        return len(slef.imgs)
