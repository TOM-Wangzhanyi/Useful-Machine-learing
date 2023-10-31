from torch.util.data import Dataset
from PIL import Image

class MyDataset(Dataset):  #继承了Dataset类，要重新三个方法
    def __init__(self,text_path,transform=None,target_transform=None): # 读入txt文件，将路径和标签分开
        img_path = open(text_path,"r")#然后设置全局变量，给getitem调用
        img=[]
        for line in img_path:
            line = line.strip().split(" ")
            img.append((line[0],int(line[1])))
        self.imgs = img   # self.imgs也是新定义的变量，目的是为了将这个局部变量传出去，作为全局变量给别的函数调用
        self.transform = transform
        self.target_transform=target_transform
    
    def __getitem__(self,index):# 根据这些路径，将img进行加载，再将img和label return出去
        img_path , label = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        if(self.transform is not None):
            img = self.transform(img)
        return img , label
    
    def __len__(self):
        return len(slef.imgs)
