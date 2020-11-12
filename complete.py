import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

from PIL import Image
from torch.autograd import Variable
from sklearn import datasets, model_selection
from torch.utils.data import DataLoader, TensorDataset
from torchsummary import summary

#CUDA_LAUNCH_BLOCKING=1

TOTAL_PATH = '.Intel_AI_96.pth'
IMAGE_PATH = 'test_glacier.jpg'
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

    #이미지 데이터 읽기 및 코드 라벨링
    img = Image.open(IMAGE_PATH).convert('RGB')
    resize_img = img.resize((INPUT_SIZE, INPUT_SIZE))

    #3채널로 가보자!
    R, G, B = resize_img.split()
    R_resize_img = np.asarray(np.float32(R)/(255.0/2.0)-1.0)
    G_resize_img = np.asarray(np.float32(G)/(255.0/2.0)-1.0)
    B_resize_img = np.asarray(np.float32(B)/(255.0/2.0)-1.0)

    RGB_resize_img = np.asarray([R_resize_img, G_resize_img, B_resize_img])

    Image_data.append(RGB_resize_img)
    print('이미지 읽어오기 완료')

    #이미지 배열화
    Image_data = np.array(Image_data, dtype='float32')
    print('이미지 배열화 완료')

    network.eval()
    result = network(torch.from_numpy(Image_data).cuda())
    print(result)
    result = torch.argmax(result, dim=1)
    print(result)
    print([key for key, value in label_dic.items() if value == result])