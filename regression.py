from numpy.lib.function_base import append
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import random

import argparse
from tqdm import tqdm

import torch.utils.data as data
from torch.utils.data import Dataset

class dataset():
    def __init__(self,data):

        self.drag = []
        self.state_index = []
        self.put = []
        self.state = []
        self.target = []
        self.rotate_list = []

        for i in range(2):
            self.drag.append(float(data[i])*10)


        for i in range(3):
            self.put.append(float(data[i + 2]))

        for i in range(4):
            if i != 3:
                self.target.append(float(data[i + 5]))
            else:
                self.rotate_list.append(float(data[i + 5]))

           

        self.state.append(self.target)
        
        for i in range(5):
            state = []
            state.append(i)
            for j in range(4):
                    state.append(float(data[i*4 + 9 + j]))
            
            self.state_index.append(state)
        # 状態の一意性が故にソートする
        self.state_index.sort(key = lambda x: x[3])

        for i in range(5):
            state = []
            for j in range(5):
                if j >= 1 and j <= 3:
                    state.append(self.state_index[i][j])
                if j == 4:
                    self.rotate_list.append(self.state_index[i][j])
                
            self.state.append(state)


    def get_dataset(self):
        
        data_array_putting = []
        data_array_position = []

        for i in range(3):
            data_array_putting .append(self.put[i])

        for i in range(6):
            for j in range(3):
                data_array_position.append(self.state[i][j])

        return np.array(data_array_putting+data_array_position)
        #return np.array(data_array_putting), np.array(data_array_position)

class test_data():
    def __init__(self,data,input):

        self.state = []
        target = []

        for i in range(3):
            target.append(float(data[29 + i]))
        
        self.state.append(target)
            
        # 入力データに合わせてソートする
        for i in range(5):
            state = []
            for j in range(3):
                state.append(float(data[input.state_index[i][0]*4 + 33 + j]))
            
            self.state.append(state)
     
    def get_testdata(self):
        
        data_array = []

        for i in range(6):
            for j in range(3):
                data_array.append(self.state[i][j])

        return np.array(data_array)    

class RegDataset(data.Dataset):
    def __init__(self, csv_dir, is_train, transform=None):
        self.csv_path = csv_dir
        self.transform = transform

        f = open(self.csv_path,'r')
        reader = csv.reader(f)
        self.x = []
        self.t = []
        #self.a = []
        count = 0
        for row in reader:
            if is_train != False and count%10==0:
                count += 1
                continue
            if is_train == False and count%10!=0:
                count += 1
                continue

            data = dataset(row)
            result = test_data(row,data)
            # rotate_list = data.rotate_list
            # drag = data.drag

            tmp_x = data.get_dataset().astype(np.float32)
            tmp_t = result.get_testdata().astype(np.float32)
            _x = torch.from_numpy(tmp_x).float()
            _t = torch.from_numpy(tmp_t).float()
            #_a = torch.from_numpy(tmp_a).float()
            self.x.append(_x)
            self.t.append(_t)
            #self.a.append(_a)
            count += 1

        #print (np.shape(self.x))
        #print (np.shape(self.t))

    def __getitem__(self, index):
        return self.x[index], self.t[index] #, self.a[index]

    def __len__(self):
        # ディレクトリ内の画像枚数を返す。
        return len(self.x)

class RegNagato(nn.Module):
    def __init__(self):
        super(RegNagato, self).__init__()
        self.relu = nn.LeakyReLU(0.01)

        self.fc1 = nn.Linear(21, 336)
        self.fc2 = nn.Linear(336, 168)
        self.fc3 = nn.Linear(168, 18)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

def eval(test_num=5):

    csv_path = 'validation.csv'
    criterion = nn.MSELoss() # 最小二乗誤差

    f = open(csv_path,'r')
    reader = csv.reader(f)
    x = []
    t = []
    for row in reader:
        data = dataset(row)
        result = test_data(row,data)
        _x = torch.from_numpy(data.get_dataset().astype(np.float32)).float()
        _t = torch.from_numpy(result.get_testdata().astype(np.float32)).float()  
        x.append(_x)
        t.append(_t)

    device = torch.device("cpu")
    net = RegNagato()
    net = net.to(device)
    net = net.eval() # 評価モードにする

    # パラメータの読み込み
    param = torch.load('model_reg_2.pth')
    net.load_state_dict(param)
    with open('result.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for i in range(test_num):
            idx = random.randint(0,len(x)-1)
            output = net (x[idx])
            
            print ("{} th validation -- ".format(i))
            print ("  Test input   -- ", x[idx].numpy())
            print ("  Prediction   -- ", output.detach().numpy())
            print ("  Ground truth -- ", t[idx].numpy())
            loss = criterion(output, t[idx])
            print ("  Loss (MSE)   -- ", loss.item())

            writer.writerow(x[idx].numpy())
            writer.writerow(output.detach().numpy())
            writer.writerow(t[idx].numpy())

def train(trainloader, testloader, batch_size, epoch, lr, weight_decay, model_path='model.pth'):

    # デバイスの設
    print('Wait...')
    device = torch.device("cuda:0") # "cpu": CPUで計算する場合，GPUを使う場合 "cuda:0"などにする．
    net = RegNagato()
    net = net.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    # optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay, amsgrad=False)
    
    print (net)

    train_loss_value=[]
    train_acc_value=[] 
    test_loss_value=[] 
    test_acc_value=[]

    # 学習の開始
    for epoch in range(epoch):
        # 訓練ステップ
        print("Epoch: {}".format(epoch+1))

        sum_loss = 0.0
        for (inputs, labels) in tqdm(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels) # 損失関数

            sum_loss += loss.item()                            #lossを加算
            #_, predicted = outputs.max(1)                      #出力の最大値の添字(予想位置)を取得
            #sum_total += labels.size(0)                        #labelの数を加算（カウント）
            #sum_correct += (predicted == labels).sum().item()  #予想位置と実際の正解を比べ,正解している数を加算

            loss.backward()     # 損失から勾配を計算
            optimizer.step()    # 勾配を用いてパラメータを更新        

        print("-- train mean loss={}".format(sum_loss*batch_size/len(trainloader.dataset)))  #loss
        train_loss_value.append(sum_loss*batch_size/len(trainloader.dataset))  #traindataのlossをグラフ描画のためにlistに保持

        sum_loss = 0.0
        # テスト（検証データ）
        for (inputs, labels) in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            sum_loss += loss.item()
            #_, predicted = outputs.max(1)
            #sum_total += labels.size(0)
            #sum_correct += (predicted == labels).sum().item()
        print("-- test  mean loss={}".format(sum_loss*batch_size/len(testloader.dataset)))
        test_loss_value.append(sum_loss*batch_size/len(testloader.dataset))

    # save model
    net = net.to('cpu') # CPUで利用するための方法
    torch.save(net.state_dict(), model_path)
    
    #'''
    epochs = range(1, len(train_loss_value) + 1)

    plt.figure(figsize=(6,6))
    plt.plot(epochs, train_loss_value)
    plt.plot(epochs, test_loss_value, c='#00ff00')
    #plt.xlim(0, epoch)
    #plt.ylim(0, 2.5)
    plt.xlabel('EPOCH')
    plt.ylabel('LOSS')
    plt.legend(['train loss', 'test loss'])
    plt.title('loss')
    plt.savefig("loss_image.png")
    plt.clf()
    #'''

def main():
    PATH_TO_CSV = 'test.csv'
    BATCH_SIZE = 16
    WEIGHT_DECAY = 0.005
    LEARNING_RATE = 0.0001
    EPOCH = 50

    # definition
    trans = torchvision.transforms.ToTensor()
    # Normalize
    trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.5,), (0.5,))])

    trainset = RegDataset(PATH_TO_CSV, is_train = True)
    testset = RegDataset(PATH_TO_CSV, is_train = False)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = BATCH_SIZE,  shuffle = False, num_workers = 2)
    testloader = torch.utils.data.DataLoader(testset, batch_size = BATCH_SIZE,  shuffle = False, num_workers = 2)
    train (trainloader, testloader, BATCH_SIZE, EPOCH, LEARNING_RATE, WEIGHT_DECAY, model_path='model_reg_2.pth')

if __name__ == '__main__':
    # main()
    eval()
