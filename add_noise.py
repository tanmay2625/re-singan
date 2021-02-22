import cv2
from SinGAN.functions import np2torch
import torch

filePath="./input.tif"
folderPath="./"

im1 = cv2.imread(filePath)
h= im1.shape[0]
w= im1.shape[1]
print(h,w)
imb= cv2.resize(im1,(w//3,h//3), interpolation= cv2.INTER_CUBIC)

cv2.imwrite(folderPath+"input_lr.jpg",imb)
#cv2.imwrite(folderPath+"lr_nearest.jpg",imnn)
exit()
im1=imb

im1 = im1/255

imt= torch.from_numpy(im1)
imt = imt.type(torch.FloatTensor)
print(imt)
alpha=0.1
imt= imt + alpha*torch.randn(imt.shape)
imt= imt.clamp(0,1)
imt= imt*255
imt = imt.int()
print(imt)
imtnp= imt.numpy()
cv2.imwrite(folderPath+"lr_noisy_"+str(alpha)+".png",imtnp)
print(im1.shape)

