import argparse
import os

import numpy as np
import torch
import torchvision


from diffusion import get_time_schedule, Posterior_Coefficients, \
    sample_from_model

from DWT_IDWT.DWT_IDWT_layer import DWT_2D, IDWT_2D
from pytorch_fid.fid_score import calculate_fid_given_paths
from score_sde.models.ncsnpp_generator_adagn import  WaveletNCSNpp
from datasets_prep.dataset import create_dataset



# %%
def sample_and_test(args):
    torch.manual_seed(args.seed)
    device = 'cuda:0'

    if args.dataset == 'celebahq_16_128':
        real_img_dir = 'pytorch_fid/celebahq128_stats.npz'
    elif args.dataset == 'celebahq_16_64':
        real_img_dir = 'pytorch_fid/celebahq64_stats.npz'
    else:
        real_img_dir = args.real_img_dir

   
    def to_range_0_1(x):
        return (x + 1.) / 2.

    args.ori_image_size = args.image_size
    args.image_size = args.current_resolution
    print(args.image_size, args.ch_mult, args.attn_resolutions)

    G_NET_ZOO = { "wavelet": WaveletNCSNpp}
    gen_net = G_NET_ZOO[args.net_type]
    print("GEN: {}".format(gen_net))


    netG = gen_net(args).to(device)
    ckpt = torch.load('/content/gdrive/MyDrive/srwavediff/saved_info/srwavediff/{}/{}/netG_{}_iteration_{}.pth'.format(
        args.dataset, args.exp, args.epoch_id, args.num_iters), map_location=device)

    # loading weights from ddp in single gpu
    for key in list(ckpt.keys()):
        ckpt[key[7:]] = ckpt.pop(key)

    netG.load_state_dict(ckpt, strict=False)
    netG.eval()



    if not args.use_pytorch_wavelet:
        iwt = IDWT_2D("haar")

    T = get_time_schedule(args, device)

    pos_coeff = Posterior_Coefficients(args, device)

    iters_needed = 50000 // args.batch_size

    save_dir = "/content/gdrive/MyDrive/srwavediff/results/{}/".format(args.exp)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if not os.path.exists(os.path.join(save_dir,'fid')):
        os.makedirs(os.path.join(save_dir,'fid'))

    # test set
    dataset = create_dataset(args)
    train_size = int(0.80 * len(dataset))  # 80% for training
    test_size = len(dataset) - train_size  # 20% for testing
      
    # Set a seed for reproducibility
    torch.manual_seed(42) 
    train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])

    # test loader
    test_data_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=0,
                                              pin_memory=True)

    # wavelet pooling
    dwt = DWT_2D("haar")
    iwt = IDWT_2D("haar")
    num_levels = int(np.log2(args.ori_image_size // args.current_resolution))

    if args.measure_time: # inference time eval
        x_t_1 = torch.randn(1, int(args.num_channels / 2),
                            args.image_size, args.image_size).to(device)
        # INIT LOGGERS
        starter, ender = torch.cuda.Event(
            enable_timing=True), torch.cuda.Event(enable_timing=True)
        repetitions = 300
        timings = np.zeros((repetitions, 1))
        # GPU-WARM-UP

        idx = 0
        for _ in range(10):
            _ = sample_from_model(
                pos_coeff, netG, args.num_timesteps, x_t_1, x_t_1, T, args)
        # MEASURE PERFORMANCE
        with torch.no_grad(): 
            for rep in range(repetitions):
                
                print("computing time, repetition number: ", rep)
                sample = next(iter(test_data_loader)) # samples of the test set to every 2 epochs 
                lr = sample['SR'] 
                index = sample['Index']

                lr = lr.to(device, non_blocking=True)
                # wavelet transform lr image
                for i in range(num_levels):
                    lrll, lrlh, lrhl, lrhh = dwt(lr)
                lrw = torch.cat([lrll, lrlh, lrhl, lrhh], dim=1) # [b, 12, h/2, w/2]
                # normalize sr_data
                lrw = lrw / 2.0  # [-1, 1]
                assert -1 <= lrw.min() < 0
                assert 0 < lrw.max() <= 1

                starter.record()
                resoluted = sample_from_model(
                    pos_coeff, netG, args.num_timesteps, x_t_1, lrw, T, args)

                resoluted *= 2
                resoluted = iwt(
                         resoluted[:, :3], resoluted[:, 3:6], resoluted[:, 6:9], resoluted[:, 9:12])
                ender.record()
                # WAIT FOR GPU SYNC
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timings[rep] = curr_time
        mean_syn = np.sum(timings) / repetitions
        std_syn = np.std(timings)
        print("Inference time: {:.2f}+/-{:.2f}ms".format(mean_syn, std_syn))
        exit(0)

    if args.compute_fid: # fid evaluation
        for i in range(95):
            with torch.no_grad():

                sample = next(iter(test_data_loader))
                lr = sample['SR'] 
                
                lr = lr.to(device, non_blocking=True)
                # wavelet transform lr image
                for z in range(num_levels):
                    lrll, lrlh, lrhl, lrhh = dwt(lr)
                lrw = torch.cat([lrll, lrlh, lrhl, lrhh], dim=1) # [b, 12, h/2, w/2]
                # normalize sr_data
                lrw = lrw / 2.0  # [-1, 1]
                assert -1 <= lrw.min() < 0
                assert 0 < lrw.max() <= 1

                x_t_1 = torch.randn(
                    args.batch_size, int (args.num_channels / 2), args.image_size, args.image_size).to(device)
                resoluted = sample_from_model(
                    pos_coeff, netG, args.num_timesteps, x_t_1, lrw, T, args)

                resoluted *= 2
                resoluted = iwt(
                         resoluted[:, :3], resoluted[:, 3:6], resoluted[:, 6:9], resoluted[:, 9:12])
                resoluted = torch.clamp(resoluted, -1, 1)

                resoluted = to_range_0_1(resoluted)  # 0-1
                for j, x in enumerate(resoluted):
                    index = i * args.batch_size + j
                    torchvision.utils.save_image(
                        x, '{}fid/{}.jpg'.format(save_dir, index))
                print('generating batch ', i)

        paths = [os.path.join(save_dir,'fid'), real_img_dir]
        print(paths)

        kwargs = {'batch_size': args.batch_size, 'device': device, 'dims': 2048}
        fid = calculate_fid_given_paths(paths=paths, **kwargs)
        print('FID = {}'.format(fid))
    else: # super-resolution without evaluations
        for iteration, sample in enumerate(test_data_loader):
            with torch.no_grad():
                print("iteration: ", iteration)
                hr = sample['HR'] 
                lr = sample['SR'] 
                index = sample['Index']


                lr = lr.to(device, non_blocking=True)
                # wavelet transform lr image
                for i in range(num_levels):
                    lrll, lrlh, lrhl, lrhh = dwt(lr)
                lrw = torch.cat([lrll, lrlh, lrhl, lrhh], dim=1) # [b, 12, h/2, w/2]
                # normalize sr_data
                lrw = lrw / 2.0  # [-1, 1]
                assert -1 <= lrw.min() < 0
                assert 0 < lrw.max() <= 1

                x_t_1 = torch.randn_like(lrw).to(device)
                resoluted = sample_from_model(
                    pos_coeff, netG, args.num_timesteps, x_t_1, lrw, T, args)


                resoluted *= 2
                if not args.use_pytorch_wavelet:
                    resoluted = iwt(
                        resoluted[:, :3], resoluted[:, 3:6], resoluted[:, 6:9], resoluted[:, 9:12])

                resoluted = (torch.clamp(resoluted, -1, 1) + 1) / 2  # 0-1

                # saving HR images 
                torchvision.utils.save_image(hr, os.path.join(
                    save_dir, 'tot_hr_id{}.png'.format(iteration)), normalize=True)
                for i, x in enumerate(hr): # save each image
                    torchvision.utils.save_image(x, os.path.join(
                        save_dir, '{}_{}_hr.png'.format(iteration, i)), normalize = True)

                
                # saving LR test set images 
                torchvision.utils.save_image(lr, os.path.join(
                    save_dir, 'tot_lr_id{}.png'.format(iteration)), normalize=True)
                for i, x in enumerate(lr): # save each image
                    torchvision.utils.save_image(x, os.path.join(
                        save_dir, '{}_{}_lr.png'.format(iteration, i)), normalize = True)

                
                #saving sr images
                torchvision.utils.save_image(
                    resoluted, os.path.join (save_dir,'tot_sr_id{}.jpg'.format(iteration)), normalize=True)
                for i, x in enumerate(resoluted): # save each image
                    torchvision.utils.save_image(x, os.path.join(
                        save_dir, '{}_{}_sr.png'.format(iteration, i)), normalize = True)
                

                print("Results are saved at tot_sr_id{}.jpg".format(iteration))
                if (iteration >= 100):
                    exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ddgan parameters')
    parser.add_argument('--seed', type=int, default=42,
                        help='seed used for initialization')
    parser.add_argument('--datadir', default='./data')
    parser.add_argument('--compute_fid', action='store_true', default=False,
                        help='whether or not compute FID')
    parser.add_argument('--measure_time', action='store_true', default=False,
                        help='whether or not measure time')
    parser.add_argument('--epoch_id', type=int, default=1000)
    parser.add_argument('--num_channels', type=int, default=12,
                        help='channel of wavelet subbands')
    parser.add_argument('--centered', action='store_false', default=True,
                        help='-1,1 scale')
    parser.add_argument('--use_geometric', action='store_true', default=False)
    parser.add_argument('--beta_min', type=float, default=0.1,
                        help='beta_min for diffusion')
    parser.add_argument('--beta_max', type=float, default=20.,
                        help='beta_max for diffusion')

    parser.add_argument('--patch_size', type=int, default=1,
                        help='Patchify image into non-overlapped patches')
    parser.add_argument('--num_channels_dae', type=int, default=128,
                        help='number of initial channels in denosing model')
    parser.add_argument('--n_mlp', type=int, default=3,
                        help='number of mlp layers for z')
    parser.add_argument('--ch_mult', nargs='+', type=int,
                        help='channel multiplier')

    parser.add_argument('--num_res_blocks', type=int, default=2,
                        help='number of resnet blocks per scale')
    parser.add_argument('--attn_resolutions', default=(16,), type=int, nargs='+',
                        help='resolution of applying attention')
    parser.add_argument('--dropout', type=float, default=0.,
                        help='drop-out rate')
    parser.add_argument('--resamp_with_conv', action='store_false', default=True,
                        help='always up/down sampling with conv')
    parser.add_argument('--conditional', action='store_false', default=True,
                        help='noise conditional')
    parser.add_argument('--fir', action='store_false', default=True,
                        help='FIR')
    parser.add_argument('--fir_kernel', default=[1, 3, 3, 1],
                        help='FIR kernel')
    parser.add_argument('--skip_rescale', action='store_false', default=True,
                        help='skip rescale')
    parser.add_argument('--resblock_type', default='biggan',
                        help='tyle of resnet block, choice in biggan and ddpm')
    parser.add_argument('--progressive', type=str, default='none', choices=['none', 'output_skip', 'residual'],
                        help='progressive type for output')
    parser.add_argument('--progressive_input', type=str, default='residual', choices=['none', 'input_skip', 'residual'],
                        help='progressive type for input')
    parser.add_argument('--progressive_combine', type=str, default='sum', choices=['sum', 'cat'],
                        help='progressive combine method.')

    parser.add_argument('--embedding_type', type=str, default='positional', choices=['positional', 'fourier'],
                        help='type of time embedding')
    parser.add_argument('--fourier_scale', type=float, default=16.,
                        help='scale of fourier transform')
    parser.add_argument('--not_use_tanh', action='store_true', default=False)
    parser.add_argument('--num_workers', type=int, default=4,
                        help='num_workers')

    # generator and training
    parser.add_argument(
        '--exp', default='experiment_cifar_default', help='name of experiment')
    parser.add_argument('--real_img_dir', default='./pytorch_fid/cifar10_train_stat.npy',
                        help='directory to real images for FID computation')

    parser.add_argument('--dataset', default='cifar10', help='name of dataset')
    parser.add_argument('--image_size', type=int, default=32,
                        help='size of image')

    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--num_timesteps', type=int, default=4)

    parser.add_argument('--cond_emb_dim', type=int, default=256)
    parser.add_argument('--t_emb_dim', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=200,
                        help='sample generating batch size')

    # wavelet GAN
    parser.add_argument("--use_pytorch_wavelet", action="store_true")
    parser.add_argument("--current_resolution", type=int, default=256)
    parser.add_argument("--net_type", default="wavelet")
    parser.add_argument("--no_use_fbn", action="store_true")
    parser.add_argument("--no_use_freq", action="store_true")
    parser.add_argument("--no_use_residual", action="store_true")

    # super resolution
    parser.add_argument('--l_resolution', type=int, default=16,
                        help='low resolution need to super_resolution')
    parser.add_argument('--h_resolution', type=int, default=128,
                        help='high resolution need to super_resolution')

    args = parser.parse_args()

    sample_and_test(args)
