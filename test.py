import torch
import numpy as np

target = np.zeros([45, 150])
target = torch.from_numpy(target)
print(target[0])
print(target.shape)