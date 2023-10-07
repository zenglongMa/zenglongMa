```python
import torch;
from torch import nn;
from torch.utils.data import DataLoader, dataset, Dataset;
from torch.utils.tensorboard import SummaryWriter;
from torchvision import transforms, datasets;
from PIL import Image;
import os;
import numpy as np
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
from torch import optim;
import time
from net.utils.graph import  Graph;
from net.st_gcn import  Model;


# transform=transforms.ToTensor();
# labels_tensor=[]
# Data_tensor=[];


device=torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# 解析pkl文件
label_path='./tools/data/NTU-RGB-D/xview/val_label.pkl';
file=open(label_path,'rb');
name_label=pickle.load(file);
labels=name_label[1];


train_label_path='./tools/data/NTU-RGB-D/xview/train_label.pkl';
train_file=open(train_label_path,'rb');
train_name_label=pickle.load(train_file);
train_labels=train_name_label[1];

#读取npy文件并显示数据
data_path='./tools/data/NTU-RGB-D/xview/val_data.npy';
Data=np.load(data_path);

train_data_path='./tools/data/NTU-RGB-D/xview/train_data.npy';
train_Data=np.load(train_data_path);

# for i in range(len(labels)):
#     labels_tensor.append(torch.tensor(labels[i]));
#     temp_tensor=torch.tensor(Data[0],dtype=torch.float32);
#     Data_tensor.append(temp_tensor)

#重写Dataset方法：
class MyDataset(Dataset):
    def __init__(self,Data,labels):
        self.Data=Data;
        self.labels=labels;

    def __getitem__(self,index):
        data=self.Data[index];
        label=self.labels[index];

        return data,label;
    def __len__(self):
        return len(self.labels);

#Dataset & Dataloader
val_dataset=MyDataset(Data,labels);
val_dataloader=DataLoader(dataset=val_dataset,batch_size=8,num_workers=0,shuffle=True);

train_dataset=MyDataset(train_Data,train_labels);
train_dataloader=DataLoader(dataset=train_dataset,batch_size=8,num_workers=0,shuffle=True);

#初始化网络
graph = {'layout': 'ntu-rgb+d', 'strategy': 'spatial', 'max_hop': 1, 'dilation': 1};
model = Model(3, 60, graph_args=graph, edge_importance_weighting=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]);
# model=model.cuda();
model=model.to(device)

#损失及优化
loss_func=nn.CrossEntropyLoss();

optimizer=optim.SGD(params=model.parameters(),lr=0.01,weight_decay=0.0001);

# loss_func=loss_func.cuda();
loss_func=loss_func.to(device)
#总迭代次数
num_epoch=100;

writer=SummaryWriter("./logs/test");

if __name__=="__main__":
    print("训练集样本总数： ",len(train_dataset));
    print("验证集样本总数： ",len(val_dataset))

    for epoch in range(num_epoch):
        #train:
        if (epoch == 35 or epoch == 55 or epoch == 65):
            for p in optimizer.param_groups:
                # optimizer.defaults['lr'] = optimizer.defaults['lr'] / 10;4
                p['lr'] = p['lr'] / 10;

        print("train----------------第{}次迭代-----------------".format(epoch + 1));

        train_total_loss=0.0;
        train_total_accuracy=0.0;

        model.train();
        start_time = time.time();
        for temp_data,temp_label in train_dataloader:
            # temp_data=temp_data.cuda();
            # temp_label=temp_label.cuda();
            temp_data=temp_data.to(device)
            temp_label=temp_label.to(device)

            optimizer.zero_grad();

            output,A = model.forward(temp_data);

            loss=loss_func(output,temp_label);

            loss.backward();

            optimizer.step();
            train_total_loss+=loss;
            accuracy=(output.argmax(1)==temp_label).sum();
            train_total_accuracy+=accuracy;
        # print("第{}次训练的邻接矩阵：".format(epoch+1))
        # print(A)

        print("第{}次迭代的train_loss:{}".format(epoch + 1, train_total_loss));
        writer.add_scalar(tag="train_total_loss", scalar_value=train_total_loss, global_step=epoch);

        print("第{}次迭代的train_acc_count:{}".format(epoch + 1, train_total_accuracy));
        writer.add_scalar(tag="train_acc_count", scalar_value=train_total_accuracy, global_step=epoch);

        print("第{}次迭代的train_acc:{}".format(epoch + 1, train_total_accuracy / len(train_dataset)));
        writer.add_scalar(tag="train_acc", scalar_value=train_total_accuracy / len(train_dataset), global_step=epoch);


        val_total_loss=0.0;
        val_total_accuracy=0.0;

        print("-----------------");

        model.eval();
        with torch.no_grad():
            for temp_val_data,temp_val_label in val_dataloader:
               # temp_val_data=temp_val_data.cuda();
               # temp_val_label=temp_val_label.cuda();

               temp_val_data=temp_val_data.to(device)
               temp_val_label=temp_val_label.to(device)

               output_val,A=model(temp_val_data);
               loss_val=loss_func(output_val,temp_val_label);
               val_total_loss += loss_val;
               accuracy = (output_val.argmax(1) ==temp_val_label).sum();

               val_total_accuracy += accuracy;
            end_time = time.time();

            print("第{}次测试的邻接矩阵：".format(epoch + 1))
            print(A)

            A=A.data.cpu().numpy();
            j=0
            for i in A:
                plt.imshow(i, cmap='terrain_r')
                plt.colorbar()
                plt.savefig('./images/{}_{}.jpg'.format(epoch,j), dpi=600, bbox_inches='tight');
                j=j+1;
                plt.show()

            print("第{}次迭代的val_loss:{}".format(epoch + 1, val_total_loss));
            writer.add_scalar(tag="val_total_loss", scalar_value=val_total_loss, global_step=epoch);

            print("第{}次迭代的val_acc_count:{}".format(epoch + 1, val_total_accuracy));
            writer.add_scalar(tag="val_total_accuracy", scalar_value=val_total_accuracy, global_step=epoch);

            print("第{}次迭代的val_acc:{}".format(epoch + 1, val_total_accuracy / len(val_dataset)));
            writer.add_scalar(tag="val_acc", scalar_value=val_total_accuracy / len(val_dataset), global_step=epoch);

            print("第{}次迭代耗时{}".format(epoch + 1, end_time - start_time));
            writer.add_scalar(tag="time", scalar_value=end_time - start_time, global_step=epoch);



    # print(output.argmax(1))               #8*60的矩阵，输出，每一行最大值的列下标
    # print(output.amax(1))
    # print(temp_label)
    # print((output.argmax(1)==temp_label).sum())
    # print((output.argmax(1)==temp_label))

```

