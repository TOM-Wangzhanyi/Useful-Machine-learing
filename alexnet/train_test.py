import dataset,model
import torch
from torchvision import transforms
import os
from torch.utils.tensorboard import SummaryWriter
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize((227,227)), #不会resize通道
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5])
])
test_transform = transforms.Compose([
    transforms.Resize((227,227)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5])
])

def train_runner(model,device,train_loader,optimizer,epoch):
    model.train()
    total = 0 
    correct = 0 
    for i,data in enumerate(train_loader):
        inputs,labels=data
        inputs,labels=inputs.to(device),labels.to(device)
        optimizer.zero_grad()
        outputs=model(inputs)
        loss=torch.nn.functional.cross_entropy(outputs,labels)
        predict=outputs.argmas(dim=1)
        total=total+labels.size(0)
        correct=correct+(predict==labels).sum().item()  #sum之后还是一个张量，需要item（）来返回一个数值
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print("epoch:{},loss:{:.4f},acc{:.2f}%".format(epoch,loss.item(),100*correct/total))
            loss.append(loss.item())
            Accuracy.append(correct/total)
    return loss.item(),correct/total
def test_runner(model,device,test_loader):
    model.eval() #关闭normalization和dropout
    correct=0.0
    test_loss=0.0
    total=0
    with torch.no_grad():  #这个with下面的代码，都不会计算梯度，加速判断速度
        for data , label in test_loader:
            data , label = data.to(device) , label.to(device)
            outputs=model(data)
            test_loss=torch.nn.functional.cross_entropy(outputs,label)
            predict = outputs.argmax(dim=1)
            total = total + label.size(0)
            correct = correct + (label == predict).sum().item()
            print("test-average_loss: {:.6f},accuracy: {:.6f}%".format(test_loss/total,100*(correct/total)))
if __name__=="__main__":
    root_dir=os.path.abspath(os.path.join(os.getcwd(),"."))

    train_txt_path = os.path.join(root_dir,'data/catVSdog/train.txt')
    test_txt_path = os.path.join(root_dir,'data/catVSdog/test.txt')

    train_data = dataset.MyDataset(text_path=train_txt_path,transform=train_transform)
    test_data=dataset.MyDataset(text_path=test_txt_path,transform=test_transform)
    
    train_loader=torch.utils.data.DataLoader(dataset=train_data,batch_size=12,shuffle=True)
    test_loader=torch.utils.data.DataLoader(dataset=test_data,batch_size=12)

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=model.AlexNet(num_classes=2).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=0.0001)

    epoch=1
    Loss = []
    Accuracy=[]

    for i in range(epoch):
        loss,accuracy=train_runner(model,device,train_loader,optimizer,epoch)
        test_runner(model,device,test_loader) #每一次训练完，都用测试集判断一次
        Loss.appen(loss)
        Accuracy.append(accuracy)
    save_path=os.path.join(root_dir,"alexnet.pth")
    torch.save(model.state_dict(),save_path)
    print("model saved successfully")
    print(Loss)
    print(Accuracy)
    writer = SummaryWriter(comment='alexnet')
    for x in range(epoch):
        writer.add_scalar('train_loss', Loss[x], x)
        writer.add_scalar('train_accuracy', Accuracy[x], x)
