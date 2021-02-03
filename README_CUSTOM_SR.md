# Re-Singan (Modified SINGAN for upscaling and denoising of document images)

## Modifications 
- Use ``` --noisy_input_name``` to pass name of noisy image (both noisy and clean image should be stored in same folder) , have a look at ```commands.txt``` for example usage of CLI arguments)
- Use ``` --custom_sr_alpha``` to pass weight of reconstruction loss.
- Use ``` --train_on_last_scale``` to train on noisy image only on last scale. (We do this if we intend to teach the generator denoising only.)
- Use ```--frozenWeight``` to pass weight of (pseudo) adverserial loss using frozen discriminator.
- Use ```--skip_training``` to skip the training on clean image. (Do this if you have already completed training on clean image and now testing on noisy image.)
- Use ```--training_name``` to give name to the current run of training. This name will be appended to all the resultant output files. It will be useful to mark certain results.
- You can find the logs of training in ```logs``` folder. The logs are currently saved as text file with a timestamp (date+time) appended at the end of their name. The log files contain seeds and values of all the arguments passed as well as saves progress of losses throughout training.
- You can regenerate a perticular result by passing the same arguments as before and passing the seed of previous result using ```--manualSeed``` argument. (This was previously not working)
- ```Output/no_SR``` will store the output after passing the original image from the top scale once. This will have same resolution as original image and can be used to detect if the generator is able to denoise.
- ```--niter ``` can be used to pass number of iterations to train the network for.
## Observations 
- Following gave best results according to our limited trials.
```
time python customSR.py --input_dir ./Input/customSR --input_name clean100.png --noisy_input_name noisy100.png --sr_factor 2 --ker_size 2 --niter 1000  --lr_g 0.001 --lr_d 0.001 --custom_sr_alpha 20 --frozenWeight 0.7                  
```
 ####  (The trials we performed were limited. Still we observed perticularly that, running less number of iterations on noisy image helped)
 - 