from skimage import io as img
from SinGAN.functions import np2torch
import torch

filePath="./Input/Images/7-12.png"
folderPath="./Input/Images/"

im1 = img.imread(filePath)
im1 = im1/255

imt= torch.from_numpy(im1)
imt = imt.type(torch.FloatTensor)
print(imt)
alpha=0.05
imt= imt + alpha*torch.randn(imt.shape)
imt= imt.clamp(0,1)
imt= imt*255
imt = imt.int()
print(imt)
imtnp= imt.numpy()
img.imsave("./Input/Images/7_12_noisy_"+str(alpha)+".png",imt)
print(im1.shape)

