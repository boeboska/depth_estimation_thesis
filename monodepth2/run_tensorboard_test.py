# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
import math

writer = SummaryWriter('monodept2/100')

for i in range(-10, 10):
    writer.add_scalar('cos_value', math.cos(i))

writer.close()

#
# tensorboard --logdir ABSOLUTE PATH
# D:\perception\thesis_bob\monodepth2\monodept2\1

