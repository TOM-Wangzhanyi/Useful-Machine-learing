import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self,num_classes=2):
        super().__init__()  #super是在__init__函数里面的，要缩进
        self.net=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=96,kernel_size=11,stride=4,padding=0),
            nn.ReLU(),
            nn.LocalResponseNorm(5,alpha=0.0001,beta=0.75,k=2),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(in_channels=96,out_channels=256,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.LocalResponseNorm(5,alpha=0.0001,beta=0.75,k=2),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(in_channels=256,out_channels=384,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384,out_channels=384,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384,out_channels=256,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256*6*6,500) ,
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(500,20),
            nn.ReLU(),
            nn.Linear(20,num_classes)
        )
    def forward(self,x):
        x = self.net(x)
        x=x.view(-1,256*6*6)
        return self.classifier(x)
