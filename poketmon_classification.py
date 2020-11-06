import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from torchvision import transforms
from PIL import Image
from torch.autograd import Variable
from sklearn import datasets, model_selection
from torch.utils.data import DataLoader, TensorDataset
from torchsummary import summary

CUDA_LAUNCH_BLOCKING=1

TOTAL_PATH = '.poketmon_AI.pth'
PATH = './train_set/'
DEVICE = 'cuda'
JPG = '.jpg'
#클래스 1:1 맵핑 (폴더 = 포켓몬 이름)
#폴더 클래스 분류 => 라벨링
Data_CLASS = []
#이름 클래스 분류 => 라벨링
Sudo_CLASS = []

#데이터 라벨링 변수
#이미지 데이터(R,G,B)
Image_data = []
#라벨 데이터(1~150)
Label_data = []

learning_rate = 0.01
batch_size = 225
epochs = 10000
CONV_IN = 1
CONV_OUT = 6

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    # 합성곱층(Output_size = ((input_size - kernel_size + 2 * padding_size)/stride_size + 1)
    self.conv1 = nn.Conv2d(in_channels=CONV_IN, out_channels=3, kernel_size=5).cuda(device=DEVICE)
    self.conv2 = nn.Conv2d(in_channels=3, out_channels=CONV_OUT, kernel_size=5).cuda(device=DEVICE)

    # 전결합층
    self.fc1 = nn.Linear(in_features=CONV_OUT*29*29, out_features=300).cuda()
    self.fc2 = nn.Linear(in_features=300, out_features=150).cuda()

  def forward(self, x):
    # 풀링층(Max_pooling = n size = n으로 나눈 값)
    x = F.max_pool2d(F.relu(self.conv1(x)).cuda(device=DEVICE), 2)
    x = F.max_pool2d(F.relu(self.conv2(x)).cuda(device=DEVICE), 2)
    x = x.view(-1, CONV_OUT*29*29)
    x = F.relu(self.fc1(x)).cuda(device=DEVICE)
    x = F.softmax(self.fc2(x)).cuda(device=DEVICE)
    return x

class Brain():
    def __init__(self):
        self.network = Net().to(device=DEVICE)
        self.optimizer = optim.RMSprop(self.network.parameters(), lr=learning_rate)
        self.criterion = torch.nn.CrossEntropyLoss().cuda(device=DEVICE)

if __name__ == '__main__':
    #모델 생성
    agent = Brain()
    summary(agent.network, input_size=(1, 128, 128), device=DEVICE)

    #데이터 클래스 읽어오기
    file_list = os.listdir(PATH)
    for dir in file_list:
        Data_CLASS.append(dir)
    print('코드 클래스 읽어오기 완료')
    print(Data_CLASS)

    #의사 클래스 읽어오기
    file_read = open('category.txt', 'r')
    for dir in file_read:
        slice = dir.strip('\n')
        Sudo_CLASS.append(slice)
    print('의사코드 클래스 읽어오기 완료')
    print(Sudo_CLASS)

    data_amount = 0
    #각 클래스 별 이미지 데이터 읽기 및 코드 라벨링
    for Code in Data_CLASS:
        dir = os.listdir(PATH+Code+'/')

        for image in dir:
            data_amount += 1
            img = Image.open(PATH+Code+'/'+image).convert('RGB')
            resize_img = img.resize((128, 128))

    #1채널로 가보자!
            R, G, B = resize_img.split()
            R_resize_img = np.asarray(np.float32(R)/255.0)
            G_resize_img = np.asarray(np.float32(G)/255.0)
            B_resize_img = np.asarray(np.float32(B)/255.0)

            Gray_resize_img = np.asarray([(R_resize_img+G_resize_img+B_resize_img)/3])
            '''np.asarray([R_resize_img, B_resize_img, G_resize_img])'''

            Image_data.append(Gray_resize_img)
            Label_data.append(int(Code.strip('N'))-1)
    print('총 데이터 개수 : ', data_amount)
    print('이미지 읽어오기 및 코드 라벨링 완료')


    #이미지, 라벨 데이터 종합 변수화 및 랜덤발생기 생성
    Image_data = np.array(Image_data, dtype='float32')
    Label_data = np.array(Label_data, dtype='int64')

    train_x, test_x, train_y, test_y = model_selection.train_test_split(Image_data, Label_data, test_size=0.1)

    train_x = torch.from_numpy(train_x).float().cuda(device=DEVICE)
    train_y = torch.from_numpy(train_y).long().cuda(device=DEVICE)

    test_x = torch.from_numpy(test_x).float().cuda(device=DEVICE)
    test_y = torch.from_numpy(test_y).long().cuda(device=DEVICE)

    train = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    print('이미지, 라벨 종합 및 랜덤발생기 생성 완료')

    for epoch in range(epochs):
        total_loss = 0
        cycle = 0
        agent.optimizer.zero_grad()
        for train_x, train_y in train_loader:
            train_x, train_y = Variable(train_x), Variable(train_y)
            output = agent.network.forward(train_x)
            loss = agent.criterion(output, train_y)
            loss.backward()
            agent.optimizer.step()
            total_loss += loss.data.item()
        if epoch % 10 == 0:
            print('Epoch : ', epoch, 'Error : ', -total_loss)

    test_x, test_y = Variable(test_x), Variable(test_y)
    result = torch.max(agent.network(test_x).data, 1)[1]
    accuracy = sum(test_y.data == result) // len(test_y.data)

    print('결과', result)
    print('정확도', accuracy)
    torch.save(agent.network, TOTAL_PATH)
