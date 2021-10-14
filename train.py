'''
https://qiita.com/mathlive/items/8e1f9a8467fff8dfd03c
https://tzmi.hatenablog.com/entry/2020/03/05/222813
'''

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

import numpy
import matplotlib.pyplot as plt    #グラフ出力用module

import argparse
from tqdm import tqdm

# ネットワークモデル
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, stride=2)

        self.conv1 = nn.Conv2d(1,16,3) # in_channels, out_channels (フィルタ数), kernel_size (フィルタサイズ), stride, padding,...
        self.conv2 = nn.Conv2d(16,32,3)

        self.fc1 = nn.Linear(32 * 5 * 5, 120) # 全結合層, input_size, output_size
        self.fc2 = nn.Linear(120, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def train(trainloader, testloader, batch_size, epoch, lr, weight_decay, model_path='model.pth'):

    # デバイスの設
    print('Wait...')
    device = torch.device("cuda:0") # "cpu": CPUで計算する場合，別のGPUを使う場合 "cuda:1"などにする．
    net = Net()
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    
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
        sum_correct = 0
        sum_total = 0

        for (inputs, labels) in tqdm(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels) # 損失関数

            sum_loss += loss.item()                            #lossを加算
            _, predicted = outputs.max(1)                      #出力の最大値の添字(予想位置)を取得
            sum_total += labels.size(0)                        #labelの数を加算（カウント）
            sum_correct += (predicted == labels).sum().item()  #予想位置と実際の正解を比べ,正解している数を加算

            loss.backward()     # 損失から勾配を計算
            optimizer.step()    # 勾配を用いてパラメータを更新            

        print("-- train mean loss={}, accuracy={}"
            .format(sum_loss*batch_size/len(trainloader.dataset), float(sum_correct/sum_total)))  #lossとaccuracy出力
        train_loss_value.append(sum_loss*batch_size/len(trainloader.dataset))  #traindataのlossをグラフ描画のためにlistに保持
        train_acc_value.append(float(sum_correct/sum_total))   #traindataのaccuracyをグラフ描画のためにlistに保持

        sum_loss = 0.0
        sum_correct = 0
        sum_total = 0

        # テスト（検証データ）
        for (inputs, labels) in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            sum_loss += loss.item()
            _, predicted = outputs.max(1)
            sum_total += labels.size(0)
            sum_correct += (predicted == labels).sum().item()
        print("-- test  mean loss={}, accuracy={}"
                .format(sum_loss*batch_size/len(testloader.dataset), float(sum_correct/sum_total)))
        test_loss_value.append(sum_loss*batch_size/len(testloader.dataset))
        test_acc_value.append(float(sum_correct/sum_total))

    # save model
    net = net.to('cpu') # CPUで利用するための方法
    torch.save(net.state_dict(), model_path)

    # net.load_state_dict(torch.load(model_path))

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

    plt.plot(epochs, train_acc_value)
    plt.plot(epochs, test_acc_value, c='#00ff00')
    #plt.xlim(0, epoch)
    #plt.ylim(0, 1)
    plt.xlabel('EPOCH')
    plt.ylabel('ACCURACY')
    plt.legend(['train acc', 'test acc'])
    plt.title('accuracy')
    plt.savefig("accuracy_image.png")
    #'''

'''
def eval():

    device = torch.device("cpu") # "cpu": CPUで計算する場合，別のGPUを使う場合 "cuda:1"などにする．
    net = Net()
    net = net.to(device)

    # パラメータの読み込み
    param = torch.load('model.pth')
    net.load_state_dict(param)
    # 評価モードにする
    net = net.eval()

    # TODO: 以下は下書き
    x = # input opencvとか？？
    output = net (x)
'''

def main():
    PATH_TO_DIR = 'MNISTDataset/data'
    BATCH_SIZE = 100
    WEIGHT_DECAY = 0.005
    LEARNING_RATE = 0.0001
    EPOCH = 100

    # definition
    trans = torchvision.transforms.ToTensor()
    # Normalize
    trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.5,), (0.5,))])

    # Thanks MNIST
    trainset = torchvision.datasets.MNIST(root = PATH_TO_DIR, train = True, download = True, transform = trans)
    testset  = torchvision.datasets.MNIST(root = PATH_TO_DIR, train = False, download = True, transform = trans)

    # https://pystyle.info/pytorch-dataloader/#outline__2
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = BATCH_SIZE,  shuffle = True, num_workers = 2)
    testloader = torch.utils.data.DataLoader(testset, batch_size = BATCH_SIZE,  shuffle = False, num_workers = 2)

    train (trainloader, testloader, BATCH_SIZE, EPOCH, LEARNING_RATE, WEIGHT_DECAY, model_path='model.pth')

if __name__ == '__main__':
    # TODO: 以下はコピペしたものを書き換えただけ
    # parser = argparse.ArgumentParser(description='Process some integers.')
    # parser.add_argument('--batch_size', metavar='N', type=int, nargs='+',
    #                    help='an integer for the accumulator')
    # parser.add_argument('--epoch', dest='accumulate', action='store_const',
    #                    const=sum, default=max,
    #                    help='sum the integers (default: find the max)')

    # args = parser.parse_args()
    # print(args.accumulate(args.integers))
    main ()

