import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import visdom

from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

time_step = 300

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')


print('Using Pytorch Version: ',torch.__version__,
      'Device: ',DEVICE)

vis = visdom.Visdom(env='LSTM+FCN_fianl') #모델 결과 시각화

class FeatureDataset(Dataset):
    def __init__(self, path,time_step):
        
        _x, _y = [], []
        file_out = pd.read_csv(path)
        x = file_out.iloc[:,0:-1].values
        y = file_out.iloc[:,-1].values
        mMscaler = MinMaxScaler()
        mMscaler.fit(x) #각 Feature마다 데이터 범위가 다르기 때문에 전처리함
        x = mMscaler.fit_transform(x)

    
        for i in range(time_step, len(y)):
                _x.append(x[i-time_step:i,:])  #time_step에 해당하도록  window 생성
                _y.append(y[i])
        
        self.x = np.array(_x)
        self.y = np.array(_y)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self,idx):
        return self.x[idx], self.y[idx]

Dataset = FeatureDataset('./data/all.csv',time_step)  
print(Dataset.x.shape) #데이터 셋 확인

x_train, x_test, y_train, y_test = train_test_split(Dataset.x,Dataset.y,test_size = 0.1,shuffle=False,random_state=34)
y_test_label = y_test #sklearn confusion_matrix에서 필요

x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train)

x_test= torch.tensor(x_test, dtype=torch.float32)
y_test = torch.tensor(y_test)

from torch.utils.data import DataLoader , TensorDataset
train_dataset = TensorDataset(x_train , y_train)
test_dataset = TensorDataset(x_test , y_test)
#트레인, 테스트 데이터 세트 분리

class BlockLSTM(nn.Module):
    def __init__(self, time_steps, num_variables, lstm_hs=128, dropout=0.8,attention=False): #본 모델에서 lstm_hs는 하이퍼파라미터로 성능을 최적화 시키기 위해 사용자가 직접 조정해야함
        super().__init__()
        self.lstm = nn.LSTM(input_size=time_steps, hidden_size=lstm_hs, num_layers=1,batch_first=True)
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, x):
        # input data (batch_size, time_step, num_variables)
        x = torch.transpose(x,2,1) # 본 프로젝트에서 참고한 논문에 따르면 dimension shuffle을 통해 더 빠른 학습이 가능하다 함
        x,_ = self.lstm(x)
        
        x = x[:,-1]
        y = self.dropout(x)
        return y


    
class BlockFCNConv(nn.Module):
    def __init__(self, in_channel=1, out_channel=64, kernel_size=8,padding = 3, momentum=0.999, epsilon=0.001, squeeze=False):
        super().__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size,padding=padding)
        self.batch_norm = nn.BatchNorm1d(num_features=out_channel, eps=epsilon, momentum=momentum)
        nn.init.kaiming_uniform_(self.conv.weight)
        self.chanel = out_channel
        self.global_pooling = nn.AvgPool1d(23-kernel_size + 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # input (batch_size,time_steps,num_variables)
        x = self.conv(x)
        x = self.batch_norm(x)
        y = self.relu(x)
        return y


    


class BlockFCN(nn.Module):
    def __init__(self, num_variables, channels=[300,128, 256, 128], kernels=[8, 5, 3],paddings=[0, 0, 0], mom=0.99, eps=0.001):
        super().__init__()
        self.conv1 = BlockFCNConv(channels[0], channels[1], kernels[0],paddings[0], momentum=mom, epsilon=eps, squeeze=True)
        self.conv2 = BlockFCNConv(channels[1], channels[2], kernels[1],paddings[1], momentum=mom, epsilon=eps, squeeze=True)
        self.conv3 = BlockFCNConv(channels[2], channels[3], kernels[2],paddings[2], momentum=mom, epsilon=eps)
        output_size = num_variables - sum(kernels) + len(kernels)
        self.global_pooling = nn.AvgPool1d(output_size)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # apply Global Average Pooling 1D
        y = self.global_pooling(x)
        y = y.squeeze()
        return y

    
    



class LSTMFCN(nn.Module):
    def __init__(self, time_steps, num_variables):
        super().__init__()
        self.lstm_block = BlockLSTM(time_steps, num_variables)
        self.fcn_block = BlockFCN(num_variables)
        self.FC = nn.Linear(128+128,4)
    def forward(self, x):
        x1 = self.lstm_block(x)
        x2 = self.fcn_block(x)
        # concatenate blocks output
        x = torch.cat([x1, x2], 1)
        # pass through Softmax activation
        y = self.FC(x)
        return y
        


def train(model, train_loader, optimizer):
        cnt_0 = 0
        cnt_1 = 0
        cnt_2 = 0
        cnt_3 = 0
        train_loss = 0
        train_correct = 0
        model.train()
        cnt = len(train_loader)
        for train_x, train_y in train_loader:
            train_x = train_x.to(DEVICE)
            train_y = train_y.to(DEVICE)
            optimizer.zero_grad()
            output = model(train_x)
            loss =  F.cross_entropy(output, train_y)
            loss.backward()
            optimizer.step()
            prediction = output.max(1, keepdim = True)[1]
            for i in range(len(prediction)):
                pre = int(prediction[i])
                if pre is 0:
                    cnt_0 +=1
                elif pre is 1:
                    cnt_1 +=1
                elif pre is 2:
                    cnt_2 +=1
                elif pre is 3:
                    cnt_3 +=1
            train_correct += prediction.eq(train_y.view_as(prediction)).sum().item()
            train_loss += loss.item()
        train_loss /= cnt
        train_accuracy = 100. * train_correct / (len(train_loader)*BATCH_SIZE)
        if epoch % 2 == 0:
            print(cnt_0, '  ', cnt_1, '  ', cnt_2,'  ', cnt_3)      
            print("[Train EPOCH: {}], \tTrain Loss: {:.4f}, \tTrain Accuracy: {:.2f} % \n".format(
                            epoch, train_loss, train_accuracy))
        return train_loss, train_accuracy



def evaluate(model, test_loader):
        test_loss = 0
        test_correct = 0
        model.eval()
        cnt_0 = 0
        cnt_1 = 0
        cnt_2 = 0
        cnt_3 = 0
        cnt = len(test_loader)
        y_test_pred = []
        with torch.no_grad():
            for test_x, test_y in test_loader:
                test_x = test_x.to(DEVICE)
                test_y = test_y.to(DEVICE)
                output = model(test_x)
                loss =  F.cross_entropy(output, test_y)
                test_loss += loss.item()
                prediction = output.max(1, keepdim = True)[1]
                for i in range(len(prediction)):
                    pre = int(prediction[i])
                    y_test_pred.append(pre)
                    if pre is 0:
                        cnt_0 +=1
                    elif pre is 1:
                        cnt_1 +=1
                    elif pre is 2:
                        cnt_2 +=1
                    elif pre is 3:
                        cnt_3 +=1
                test_correct += prediction.eq(test_y.view_as(prediction)).sum().item()
            test_loss /= cnt
            test_accuracy = 100. * test_correct / (len(test_loader)*BATCH_SIZE)
            if epoch % 2 == 0:
                print(cnt_0, '  ', cnt_1, '  ', cnt_2,'  ', cnt_3)
                print("[EPOCH: {}], \tTest Loss: {:.4f}, \tTest Accuracy: {:.2f} % \n".format(
                                epoch, test_loss, test_accuracy))
                print(classification_report(y_test_label, y_test_pred, target_names=['class 0', 'class 1','class 2','class 3']))

        return test_loss, test_accuracy
    
                
        
EPOCHS = 100
batch_size = [128]
learning_rates = [0.001]

from torchsampler import ImbalancedDatasetSampler
for BATCH_SIZE in batch_size:
    for learning_rate in learning_rates:
        model = LSTMFCN(300,23).to(DEVICE)
        train_loader = torch.utils.data.DataLoader(train_dataset,batch_size = BATCH_SIZE,
                                                     shuffle=True)
 
        tit = '_' + str(BATCH_SIZE) +'_' + str(learning_rate)
        
        
        
        loss_plt = vis.line(Y = torch.Tensor(1,2).zero_(),
                            opts = dict(title = 'loss'+tit, legend = ['train_loss','test_loss'],
                                        showlegend=True))  

        accuracy_plt = vis.line(Y = torch.Tensor(1,2).zero_(),
                            opts = dict(title = 'accuracy'+tit, legend = ['train_accuracy','test_accuracy'],
           showlegend=True))
        
        
        
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = False
                                                  )
        
        optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
            
        print("BATCH_SIZE : ",BATCH_SIZE, "lr = ",learning_rate)
        
            #if epoch % 10 == 0:
            #    learning_rate = learning_rate *0.5
            
        for epoch in range(EPOCHS):
            train_loss, train_accuracy = train(model, train_loader, optimizer)
            loss, accuracy = evaluate(model, test_loader)
            loss = torch.Tensor([[train_loss,loss]])
            accuracy = torch.Tensor([[train_accuracy,accuracy]])
            vis.line(X = torch.Tensor([epoch]), Y = loss, win=loss_plt,update = 'append')
            vis.line(X = torch.Tensor([epoch]), Y = accuracy, win=accuracy_plt,update = 'append')
            
            
            if epoch % 2 == 0:
                print('--------------------------')
                
                








