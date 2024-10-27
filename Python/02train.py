import torch
from PIL import Image
from torchvision import datasets, models, transforms,utils
import torch.nn as nn
import numpy as np
import random
import os
import torchvision
from tqdm import tqdm
from PIL import ImageFile


def setup_seed(seed):
   torch.manual_seed(seed)
   torch.cuda.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   np.random.seed(seed)
   random.seed(seed)
   torch.backends.cudnn.benchmark = False
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.enabled = True
   os.environ['PYTHONHASHSEED'] = str(seed)
setup_seed(20)

root = './'

# Hyper parameters
num_epochs = 10   
batch_size = 2
learning_rate = 0.00005  
momentum = 0.9  
num_classes = 94 


class MyDataset(torch.utils.data.Dataset):  
    def __init__(self, datatxt, transform=None, target_transform=None): 
        super(MyDataset, self).__init__()
        fh = open(datatxt, 'r')  
        imgs = [] 
        for line in fh: 
            line2 = line.rstrip() 
            words = line2.split()  
            imgs.append((line[:-3], int(words[-1]))) 
    
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):  
        fn, label = self.imgs[index]  
        img = Image.open(fn).convert('RGB')  
        img = img.resize((224,224))

        if self.transform is not None:
            img = self.transform(img) 
        return img, label  

    def __len__(self):  
        return len(self.imgs)

train_data = MyDataset(datatxt=root + 'train.txt', transform=transforms.ToTensor())
test_data = MyDataset(datatxt=root + 'test.txt', transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size = batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size = batch_size, shuffle=False)

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class MobileNet(nn.Module):
    def __init__(self, num_classes=num_classes):   
        super(MobileNet, self).__init__()
        net = models.mobilenet_v2(pretrained=True)
        net.classifier = nn.Sequential()
        self.features = net
        self.classifier = nn.Sequential(   
                nn.Linear(1280, 1000),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(1000, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

net = MobileNet().to(device)

criterion = nn.CrossEntropyLoss()

optimizer=torch.optim.Adam(net.parameters(),lr=learning_rate,betas=(0.9,0.999))

ImageFile.LOAD_TRUNCATED_IMAGES = True
# train_accs = []
# train_loss = []
test_acc2 = []
test_loss2 = []
# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    running_loss = 0.0
    train_loader = tqdm(train_loader)
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # Forward pass
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    net.eval()
    test_loss = 0.
    test_acc = 0.
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            # batch_x, batch_y = Variable(batch_x, volatile=True), Variable(batch_y, volatile=True)
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            out = net(batch_x)
            loss2 = criterion(out, batch_y)
            test_loss += loss2.item()
            pred = torch.max(out, 1)[1]
            num_correct = (pred == batch_y).sum()
            test_acc += num_correct.item()
        test_acc2.append(test_acc/len(test_data))
        test_loss2.append(test_loss/len(test_data))
        print('Epoch :{}, Test Loss: {:.6f}, Acc: {:.6f}'.format(epoch + 1, test_loss / (len(
            test_data)), test_acc / (len(test_data))))


    #torch.save(net, './logs/model6.ckpt')
    # Export to ONNX format
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    onnx_path = './logs/model7.onnx'
    #torch.onnx.export(net, dummy_input, onnx_path, opset_version=11)
    torch.onnx.export(net, dummy_input, onnx_path,
                      opset_version=11,
                      input_names=['input'],  # 这里指定你希望的输入名称
                      output_names=['output'])  # 也可以指定输出名称



    print("ONNX model exported at:", onnx_path)
print(test_acc2)


