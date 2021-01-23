from argparse import Namespace
from math import fabs
import sys
from skimage import io as img
import SinGAN.functions as functions
import torch.nn as nn

im1path= sys.argv[1]
im2path= sys.argv[2]
print(im1path,im2path)
opt= Namespace()
opt.__setattr__('nc_im',3)
opt.__setattr__('not_cuda',False)
im1 = functions.np2torch(
    img.imread(
        im1path
    ),opt
)
im2 = functions.np2torch(
    img.imread(
        im2path
    ),opt
)
#print(im1,im2)
loss= nn.MSELoss()

loss_v= loss(im1,im2)

print(loss_v)
