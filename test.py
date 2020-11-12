import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from PIL import Image
from torch.autograd import Variable
from sklearn import datasets, model_selection
from torch.utils.data import DataLoader, TensorDataset
from torchsummary import summary

#CUDA_LAUNCH_BLOCKING=1

TOTAL_PATH = '.Intel_AI_96.pth'
PATH = './Intel_archive/seg_train/'
DEVICE = 'cuda'
JPG = '.jpg'
#클래스 코드(Dictionary)
label_dic = {
        'buildings' : 0,
        'forest' : 1,
        'glacier' : 2,
        'mountain' : 3,
        'sea' : 4,
        'street' : 5
    }

#클래스 1:1 맵핑 (폴더 = 포켓몬 이름)
#폴더 클래스 분류 => 라벨링
Data_CLASS = []
#이름 클래스 분류 => 라벨링
Sudo_CLASS = []

#데이터 라벨링 변수
#이미지 데이터(R,G,B)
Image_data = []
#라벨 데이터(0,1,2,3,4,5)
Label_data = []
learning_rate = 0.01
batch_size = 128
epochs = 1000
INPUT_SIZE = 32
KERNEL_SIZE = 5
CONV_IN = 3
CONV_MID1 = 22
CONV_OUT = 28
FC_IN = 5

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    # 합성곱층(Output_size = ((input_size - kernel_size + 2 * padding_size)/stride_size + 1)
    self.conv1 = nn.Conv2d(in_channels=CONV_IN, out_channels=CONV_MID1, kernel_size=KERNEL_SIZE).cuda(device=DEVICE)
    self.conv2 = nn.Conv2d(in_channels=CONV_MID1, out_channels=CONV_OUT, kernel_size=KERNEL_SIZE).cuda(device=DEVICE)

    # 전결합층
    self.fc1 = nn.Linear(in_features=CONV_OUT*FC_IN*FC_IN, out_features=12).cuda()
    self.fc2 = nn.Linear(in_features=12, out_features=6).cuda()

  def forward(self, x):
    # 풀링층(Max_pooling = n size = n으로 나눈 값)
    x = F.max_pool2d(F.relu(self.conv1(x)).cuda(device=DEVICE), 2)
    x = F.max_pool2d(F.relu(self.conv2(x)).cuda(device=DEVICE), 2)
    x = x.view(-1, CONV_OUT*FC_IN*FC_IN)
    x = F.relu(self.fc1(x)).cuda(device=DEVICE)
    x = F.log_softmax(self.fc2(x), dim=1).cuda(device=DEVICE)
    return x

if __name__ == '__main__':
    #모델 생성
    network = torch.load(TOTAL_PATH)
    optimizer = optim.RMSprop(network.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss().cuda(device=DEVICE)
    summary(network, input_size=(3, INPUT_SIZE, INPUT_SIZE), device=DEVICE)

    #데이터 클래스 읽어오기
    file_list = os.listdir(PATH)
    for dir in file_list:
        Data_CLASS.append(dir)
    print('코드 클래스 읽어오기 완료')
    print(Data_CLASS)

    data_amount = 0
    #각 클래스 별 이미지 데이터 읽기 및 코드 라벨링
    for Code in Data_CLASS:
        dir = os.listdir(PATH+Code)

        for image in dir:
            data_amount += 1
            img = Image.open(PATH+Code+'/'+image).convert('RGB')
            resize_img = img.resize((INPUT_SIZE, INPUT_SIZE))

            #1채널로 가보자!
            R, G, B = resize_img.split()
            R_resize_img = np.asarray(np.float32(R)/(255.0/2.0)-1.0)
            G_resize_img = np.asarray(np.float32(G)/(255.0/2.0)-1.0)
            B_resize_img = np.asarray(np.float32(B)/(255.0/2.0)-1.0)

            RGB_resize_img = np.asarray([R_resize_img, G_resize_img, B_resize_img])

            Image_data.append(RGB_resize_img)
            Label_data.append(label_dic[Code])
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

    test = TensorDataset(test_x, test_y)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=True)
    print('이미지, 라벨 종합 및 랜덤발생기 생성 완료')

    total = batch_size
    correct = 0
    Avg_success = []
    network.eval()
    with torch.no_grad():
        for epoch in range(epochs):
            for test_x, test_y in test_loader:
                test_x, test_y = Variable(test_x), Variable(test_y)
                output = network(test_x)
                predict = torch.argmax(output, dim=1)
                correct = torch.eq(predict, test_y)
                correct = sum(correct)
                percentage = int(correct)/int(total)
                print(epoch, ' percentage : ', percentage)
                Avg_success.append(percentage)
    print('AVG percentage : {}'.format(sum(Avg_success)/len(Avg_success)))


