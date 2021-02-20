import SinGAN.functions as functions
from numpy.core.fromnumeric import shape
from skimage import io as img
import subprocess
import os

cnt = 0

for im_name in os.listdir("./train_cleaned/"):
    command = (
        "time python customSR.py --input_dir ./train_resized_clean/ --input_name lr_2_cubic.jpg --noisy_input_name %s --sr_factor 2 --ker_size 2 --niter 1000  --lr_g 0.001 --lr_d 0.001 --custom_sr_alpha 20 --frozenWeight 0.7 --skip_training 1"
    ) % (im_name[:-4] + "_noisy.png")
    subprocess.run(
        [
            "python",
            "customSR.py",
            "--input_dir",
            "./train_resized_clean/",
            "--input_name",
            "lr_2_cubic.jpg",
            "--noisy_input_name",
            im_name[:-4] + "_noisy.png",
            "--sr_factor",
            "2",
            "--ker_size",
            "2",
            "--niter",
            "1000",
            "--lr_g",
            "0.001",
            "--lr_d",
            "0.001",
            "--custom_sr_alpha",
            "20",
            "--frozenWeight",
            "0.7",
            "--skip_training",
            "1"
        ]
    )
    cnt += 1
    if cnt == 10:
        exit()
