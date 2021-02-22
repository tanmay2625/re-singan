import SinGAN.functions as functions
from numpy.core.fromnumeric import shape
from skimage import io as img
import subprocess
import os

from torch import sub

#name_list= os.listdir("./train_resized_clean/")

name_list=[
    '122.png',
    '158.png',
    '38.png',
    '63.png',
    '2.png',
    '141.png',
    '132.png',
    '8.png',
    '21.png',
    '29.png',
    '99.png',
    '87.png',
    '56.png',
    '95.png',
    '14.png',
    '72.png',
    '150.png',
    '47.png',
    '81.png'
]
name_list=sorted(name_list)
print(name_list)



exit()
for im_name in name_list:
    if im_name[-9:-4]=="noisy": continue
    
    # subprocess.run(
    #     [
    #         "python",
    #         "test_sin.py",
    #         "--input_dir",
    #         "./train_resized_clean/",
    #         "--input_name",
    #         im_name,
    #         "--noisy_input_name",
    #         im_name,
    #         "--sr_factor",
    #         "2",
    #         "--ker_size",
    #         "2",
    #         "--niter",
    #         "1000",
    #         "--custom_sr_alpha",
    #         "100",
    #         "--frozenWeight",
    #         "0.7",
    #         "--skip_training",
    #         "1"
    #     ]
    # )

    print(im_name)
    subprocess.run([
        "python",
        "metrics.py",
        "--im1",
        "./train_cleaned/"+im_name,
        "--im2",
        "Output/SR/2.0/"+im_name[:-4]+"_HR_test.png"
    ])
    #exit()