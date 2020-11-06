import math
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()
for step in range(-360,360):
    angle_rad = step * math.pi / 180
    writer.add_scalar('sin', math.sin(angle_rad),step)
    writer.add_scalar('cos', math.cos(angle_rad),step)

writer.close()