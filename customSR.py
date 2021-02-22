from ast import parse
from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training import *
from SinGAN.imresize import imresize
import SinGAN.functions as functions
from SinGAN.logger import *

def saveUpScaledImage(imageName,deepFreeze=0):
    print('%f' % pow(in_scale, iter_num))
    logger.log_('Super Res by %f'%pow(in_scale, iter_num))
    global Gs
    Zs_sr = []
    reals_sr = []
    NoiseAmp_sr = []
    Gs_sr = []

    real = functions.np2torch( img.imread('%s/%s'%(opt.input_dir,imageName)),opt )    #reals[-1]  # read_image(opt)
    real_ = real
    opt.scale_factor = 1 / in_scale
    opt.scale_factor_init = 1 / in_scale
    for j in range(1, iter_num + 1, 1):
        real_ = imresize(real_, pow(1 / opt.scale_factor, 1), opt)
        reals_sr.append(real_)
        Gs_sr.append(Gs[-1])
        NoiseAmp_sr.append(NoiseAmp[-1])
        z_opt = torch.full(real_.shape, 0, device=opt.device)
        m = nn.ZeroPad2d([3,2,3,2])
        z_opt = m(z_opt)
        Zs_sr.append(z_opt)
    
    #only denoising
    m = nn.ZeroPad2d(tuple([3,2,3,2]))
    real=m(real)
    real_out= Gs[-1](real.detach(), real.detach())
    if len(imageName): plt.imsave('%s/%s_%s_%s_seed=%d_orig_res.png' % ('Output/no_SR/',imageName[:-4],opt.tx,opt.training_name,opt.manualSeed), functions.convert_image_np(real_out.detach()), vmin=0, vmax=1)

    out = SinGAN_generate(Gs_sr, Zs_sr, reals_sr, NoiseAmp_sr, opt, in_s=reals_sr[0], num_samples=1, imageName=imageName)
    out = out[:, :, 0:int(opt.sr_factor * reals[-1].shape[2]), 0:int(opt.sr_factor * reals[-1].shape[3])]
    dir2save = functions.generate_dir2save(opt,deepFreeze)
    
    plt.imsave('%s/%s_HR.png' % (dir2save,imageName[:-4]), functions.convert_image_np(out.detach()), vmin=0, vmax=1)

def trainOnClean():
    opt.mode = 'train'    
    print('*** Train SinGAN for SR on clean image***')
    global Gs,Ds,Zs,reals,NoiseAmp
    Gs = []
    Ds = []
    Zs = []
    reals = []
    NoiseAmp = []
    tempp= opt.train_on_last_scale
    opt.train_on_last_scale= 0
    trainCustom(opt, Gs, Zs,Ds, reals, NoiseAmp)
    print(len(Gs))
    opt.train_on_last_scale=tempp
    opt.mode=mode
    saveUpScaledImage(opt.input_name)

def trainOnNoisy():
    opt.mode='train'
    global Gs,Ds,Zs,reals,NoiseAmp
    Gs = []
    Ds = []
    Zs = []
    reals = []
    NoiseAmp = []
    trainCustom(opt,Gs,Zs, Ds,reals,NoiseAmp,deepFreeze=1)
    opt.mode = mode
    saveUpScaledImage(opt.noisy_input_name,1)

if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--input_dir', help='input image dir', default='Input/Images')
    parser.add_argument('--input_name', help='training image name', default="33039_LR.png")#required=True)
    parser.add_argument('--noisy_input_name', help='training image name', default="33039_LR.png")
    parser.add_argument('--sr_factor', help='super resolution factor', type=float, default=4)
    parser.add_argument('--mode', help='task to be done', default='SR')
    parser.add_argument('--custom_sr_alpha',help='alpha for custom sr',type=int,default=100)
    parser.add_argument('--train_on_last_scale',help='train noisy image exclusively on last scale',type=int,default=0)
    parser.add_argument('--frozenWeight',help='weight for adverserial loss by frozen discriminator',type=float,default=1)
    parser.add_argument('--training_name',help='add name to the training',type=str,default='')
    parser.add_argument('--skip_training',help='skips training on clean image',type=int,default=0)
    parser.add_argument('--tx',help='timstamp',default='')
    opt = parser.parse_args()
    opt = functions.post_config(opt)
    print(type(opt.custom_sr_alpha),opt.custom_sr_alpha)
    opt.alpha=opt.custom_sr_alpha
    logger.initiate(opt)
    logger.log_('seed-> %d'%(opt.manualSeed))
    x=datetime.datetime.today()
    x= x.strftime("%b-%d-%Y-%H:%M:%S")
    x=x[-8:] #time of starting
    opt.tx=x

    logger.log_(opt.__repr__())
    Gs = []
    Zs = []
    Ds=[]
    reals = []
    NoiseAmp = []
    dir2save = functions.generate_dir2save(opt)
    
    if dir2save is None:
        print('task does not exist')
    #elif (os.path.exists(dir2save)):
    #    print("output already exist")
    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass

        mode = opt.mode
        in_scale, iter_num = functions.calc_init_scale(opt)
        opt.scale_factor = 1 / in_scale
        opt.scale_factor_init = 1 / in_scale
        real = functions.read_image(opt)
        opt.min_size = 18
        real = functions.adjust_scales2image_SR(real, opt)

        Gs = []
        Ds = []
        Zs = []
        reals = []
        NoiseAmp = []
        
        if not opt.skip_training: trainOnClean()
        trainOnNoisy()
