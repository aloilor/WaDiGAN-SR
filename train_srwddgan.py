import argparse
import os
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from datasets_prep.dataset import create_dataset
from diffusion import sample_from_model, sample_posterior, \
    q_sample_pairs, get_time_schedule, \
    Posterior_Coefficients, Diffusion_Coefficients
from DWT_IDWT.DWT_IDWT_layer import DWT_2D, IDWT_2D
from pytorch_wavelets import DWTForward, DWTInverse
from torch.multiprocessing import Process
from utils import init_processes, copy_source, broadcast_params


def grad_penalty_call(args, D_real, x_t):
    grad_real = torch.autograd.grad(
        outputs=D_real.sum(), inputs=x_t, create_graph=True
    )[0]
    grad_penalty = (
        grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
    ).mean()

    grad_penalty = args.r1_gamma / 2 * grad_penalty
    grad_penalty.backward()


# %%
def train(rank, gpu, args):
    from EMA import EMA
    from score_sde.models.discriminator import Discriminator_large, Discriminator_small
    from score_sde.models.ncsnpp_generator_adagn import WaveletNCSNpp

    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    device = torch.device('cuda:{}'.format(gpu))

    batch_size = args.batch_size


    # experiment path creation
    exp = args.exp
    parent_dir = "/content/gdrive/MyDrive/srwavediff/saved_info/srwavediff/{}".format(args.dataset)

    exp_path = os.path.join(parent_dir, exp)
    if rank == 0:
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)
            copy_source(__file__, exp_path)
            shutil.copytree('score_sde/models',
                            os.path.join(exp_path, 'score_sde/models'))

    # train set and test set
    dataset = create_dataset(args)

    # 0.8;0.2 for celebA
    # 0.9;0.1 for df2k
    train_size = int(0.90 * len(dataset))  
    test_size = len(dataset) - train_size  

    # Set a seed for reproducibility
    torch.manual_seed(42) 
    train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])

    # train set loader and sampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set,
                                                                    num_replicas=args.world_size,
                                                                    rank=rank)
    train_data_loader = torch.utils.data.DataLoader(train_set,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=args.num_workers,
                                              pin_memory=True,
                                              sampler=train_sampler,
                                              drop_last=True)

    # test loader
    test_data_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=args.num_workers,
                                              pin_memory=True)


    test_samples = next(iter(test_data_loader)) # samples of the test set to every 2 epochs 
    test_hr_image = test_samples['HR'] # test hr_img
    test_sr_image = test_samples['SR'] # sr_img - bicubic interpolated
    # saving HR test set images 
    torchvision.utils.save_image(test_hr_image, os.path.join(
        exp_path, 'test_hr.png'), normalize=True)
    # saving LR test set images 
    torchvision.utils.save_image(test_sr_image, os.path.join(
        exp_path, 'test_lr.png'), normalize=True)


    args.ori_image_size = args.image_size
    args.image_size = args.current_resolution
    G_NET_ZOO = {"wavelet": WaveletNCSNpp}
    gen_net = G_NET_ZOO[args.net_type]
    disc_net = [Discriminator_small, Discriminator_large]
    netG = gen_net(args).to(device)

    if args.dataset in ['celebahq_16_64']:
        # small images disc
        print("Small images disc\n")
        netD = disc_net[0](nc=args.num_channels, ngf=args.ngf,
                           t_emb_dim=args.t_emb_dim,
                           act=nn.LeakyReLU(0.2), num_layers=args.num_disc_layers).to(device)
    elif args.dataset in ['celebahq_16_128', 'div2k_128_512', 'df2k_128_512']:
        # large images disc
        print("GEN: {}, DISC: large images disc\n".format(gen_net))
        netD = disc_net[1](nc=args.num_channels, ngf=args.ngf,
                           t_emb_dim=args.t_emb_dim,
                           act=nn.LeakyReLU(0.2), num_layers=args.num_disc_layers).to(device)


    broadcast_params(netG.parameters())
    broadcast_params(netD.parameters())

    optimizerD = optim.Adam(filter(lambda p: p.requires_grad, netD.parameters(
    )), lr=args.lr_d, betas=(args.beta1, args.beta2))
    optimizerG = optim.Adam(filter(lambda p: p.requires_grad, netG.parameters(
    )), lr=args.lr_g, betas=(args.beta1, args.beta2))

    if args.use_ema:
        optimizerG = EMA(optimizerG, ema_decay=args.ema_decay)

    schedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizerG, args.num_epoch, eta_min=1e-5)
    schedulerD = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizerD, args.num_epoch, eta_min=1e-5)

    # ddp
    netG = nn.parallel.DistributedDataParallel(
        netG, device_ids=[gpu], find_unused_parameters=True)
    netD = nn.parallel.DistributedDataParallel(netD, device_ids=[gpu])

    # Wavelet Pooling
    if not args.use_pytorch_wavelet:
        dwt = DWT_2D("haar")
        iwt = IDWT_2D("haar")
    else:
        dwt = DWTForward(J=1, mode='zero', wave='haar').cuda()
        iwt = DWTInverse(mode='zero', wave='haar').cuda()
    num_levels = int(np.log2(args.ori_image_size // args.current_resolution))

    test_sr = test_sr_image.to(device, non_blocking=True)
    # wavelet transform lr test_set images
    if not args.use_pytorch_wavelet:
        for i in range(num_levels):
            test_srll, test_srlh, test_srhl, test_srhh = dwt(test_sr)
    else:
        test_srll, test_srh = dwt(test_sr)  # [b, 3, h, w], [b, 3, 3, h, w]
        test_srlh, test_srhl, test_srhh = torch.unbind(test_srh[0], dim=2)
    test_sr_data = torch.cat([test_srll, test_srlh, test_srhl, test_srhh], dim=1) # [b, 12, h/2, w/2]

    # normalize sr_data
    test_sr_data = test_sr_data / 2.0  # [-1, 1]

    assert -1 <= test_sr_data.min() < 0
    assert 0 < test_sr_data.max() <= 1

    coeff = Diffusion_Coefficients(args, device)
    pos_coeff = Posterior_Coefficients(args, device)
    T = get_time_schedule(args, device)

    if args.resume or os.path.exists(os.path.join(exp_path, 'content.pth')):
        checkpoint_file = os.path.join(exp_path, 'content.pth')
        checkpoint = torch.load(checkpoint_file, map_location=device)
        init_epoch = checkpoint['epoch']
        epoch = init_epoch
        # load G
        netG.load_state_dict(checkpoint['netG_dict'])
        optimizerG.load_state_dict(checkpoint['optimizerG'])
        schedulerG.load_state_dict(checkpoint['schedulerG'])
        # load D
        netD.load_state_dict(checkpoint['netD_dict'])
        optimizerD.load_state_dict(checkpoint['optimizerD'])
        schedulerD.load_state_dict(checkpoint['schedulerD'])

        global_step = checkpoint['global_step']
        print("=> loaded checkpoint (epoch {}, iteration{})"
              .format(checkpoint['epoch'],checkpoint['global_step']))
        
        # clean the gpu to make sure we don't run OOM
        del checkpoint
        torch.cuda.empty_cache()
        torch.cuda.empty_cache() 
    else:
        global_step, epoch, init_epoch = 0, 0, 0


    print("Starting training loop. \n")
    for epoch in range(init_epoch, args.num_epoch + 1):
        train_sampler.set_epoch(epoch)

        for iteration, data_dict in enumerate(train_data_loader):
            
            # update disc
            for p in netD.parameters():
                p.requires_grad = True
            netD.zero_grad()
            
            #generator frozen
            for p in netG.parameters():
                p.requires_grad = False

            hr_image = data_dict['HR'] # hr_img
            sr_image = data_dict['SR'] # sr_img - bicubic interpolated

            # sample from p(x_0)
            x0 = hr_image.to(device, non_blocking=True)

            # wavelet transform x0 - hr images
            if not args.use_pytorch_wavelet:
                for i in range(num_levels):
                    xll, xlh, xhl, xhh = dwt(x0) # [b, 3, h, w], [b, 3, 3, h, w]
            else:
                xll, xh = dwt(x0)  # [b, 3, h, w], [b, 3, 3, h, w]
                xlh, xhl, xhh = torch.unbind(xh[0], dim=2)
            real_data = torch.cat([xll, xlh, xhl, xhh], dim=1)  # [b, 12, h/2, w/2]

            # normalize real_data
            real_data = real_data / 2.0  # [-1, 1]

            assert -1 <= real_data.min() < 0
            assert 0 < real_data.max() <= 1

            # sr images
            sr = sr_image.to(device, non_blocking=True)
            
            # wavelet transform sr images
            if not args.use_pytorch_wavelet:
                for i in range(num_levels):
                    srll, srlh, srhl, srhh = dwt(sr)
            else:
                srll, srh = dwt(sr)  # [b, 3, h, w], [b, 3, 3, h, w]
                srlh, srhl, srhh = torch.unbind(srh[0], dim=2) 
            sr_data = torch.cat([srll, srlh, srhl, srhh], dim=1) # [b, 12, h/2, w/2]

            # normalize sr_data
            sr_data = sr_data / 2.0  # [-1, 1]
            assert -1 <= sr_data.min() < 0
            assert 0 < sr_data.max() <= 1

            # sample t
            t = torch.randint(0, args.num_timesteps,
                              (real_data.size(0),), device=device)

            x_t, x_tp1 = q_sample_pairs(coeff, real_data, t)
            x_t.requires_grad = True

            # train with real
            D_real = netD(x_t, t, x_tp1.detach()).view(-1)
            errD_real = F.softplus(-D_real).mean()

            errD_real.backward(retain_graph=True)

            if args.lazy_reg is None:
                grad_penalty_call(args, D_real, x_t)
            else:
                if global_step % args.lazy_reg == 0:
                    grad_penalty_call(args, D_real, x_t)

            # train with fake
            x_0_predict = netG(x_tp1.detach(), t, sr_data)

            x_pos_sample = sample_posterior(pos_coeff, x_0_predict, x_tp1, t) # x(t-1) fake 

            output = netD(x_pos_sample, t, x_tp1.detach()).view(-1)
            errD_fake = F.softplus(output).mean()

            errD_fake.backward()

            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            # frozen disc
            for p in netD.parameters():
                p.requires_grad = False

            # update gen
            for p in netG.parameters():
                p.requires_grad = True
            netG.zero_grad()

            t = torch.randint(0, args.num_timesteps,
                              (real_data.size(0),), device=device)
            x_t, x_tp1 = q_sample_pairs(coeff, real_data, t)

            x_0_predict = netG(x_tp1.detach(), t, sr_data)
            x_pos_sample = sample_posterior(pos_coeff, x_0_predict, x_tp1, t)

            output = netD(x_pos_sample, t, x_tp1.detach()).view(-1)
            errG = F.softplus(-output).mean()

            # reconstruction loss
            if args.rec_loss:
                rec_loss = F.l1_loss(x_0_predict, real_data)
                errG = errG + rec_loss

            errG.backward()
            optimizerG.step()

            global_step += 1

            if global_step % 100 == 0 and global_step != 0:
                if rank == 0:
                    
                    loss_file = open('{}/losses.txt'.format(exp_path), 'a') # saving losses in it 
                    loss_file.write('epoch {} iteration{}, G Loss: {}, D Loss: {}\n'.format(
                        epoch, global_step, errG.item(), errD.item()))
                    loss_file.close()
                    print('epoch {} iteration{}, G Loss: {}, D Loss: {}'.format(
                        epoch, global_step, errG.item(), errD.item()))

            if rank == 0:
                #save content
                if args.save_content:
                    if global_step % args.save_content_every == 0:
                        content = {'epoch': epoch, 'global_step': global_step, 'args': args,
                                'netG_dict': netG.state_dict(), 'optimizerG': optimizerG.state_dict(),
                                'schedulerG': schedulerG.state_dict(), 'netD_dict': netD.state_dict(),
                                'optimizerD': optimizerD.state_dict(), 'schedulerD': schedulerD.state_dict()}
                        torch.save(content, os.path.join(exp_path, 'content.pth'))
                        print('Content saved.')
                # save checkpoint
                if global_step % args.save_ckpt_every == 0:
                    if args.use_ema:
                        optimizerG.swap_parameters_with_ema(
                            store_params_in_ema=True)

                    torch.save(netG.state_dict(), os.path.join(
                        exp_path, 'netG_{}_iteration_{}.pth'.format(epoch,global_step)))
                    if args.use_ema:
                        optimizerG.swap_parameters_with_ema(
                            store_params_in_ema=True)
                    print("checkpoint saved.")

                # inference round
                if global_step % args.save_content_every == 0:
                    with torch.no_grad():
                        # saving SR images 
                        torchvision.utils.save_image(sr_image, os.path.join(
                            exp_path, 'sr_epoch_{}_iteration_{}.png'.format(epoch,global_step)), normalize=True)

                        # saving posterior samples 
                        x_pos_sample = x_pos_sample[:, :3]
                        torchvision.utils.save_image(x_pos_sample, os.path.join(
                            exp_path, 'xpos_epoch_{}_iteration_{}.png'.format(epoch,global_step)), normalize=True)

                        # inference on test batch
                        x_t_1 = torch.randn_like(real_data)
                        resoluted = sample_from_model(
                            pos_coeff, netG, args.num_timesteps, x_t_1, test_sr_data, T, args)

                        #inference on train batch
                        resoluted_train = sample_from_model(
                            pos_coeff, netG, args.num_timesteps, x_t_1, sr_data, T, args)

                        x_0_predict *= 2
                        real_data *= 2
                        resoluted *= 2
                        resoluted_train *= 2
                        if not args.use_pytorch_wavelet:
                            x_0_predict = iwt(
                                x_0_predict[:, :3], x_0_predict[:, 3:6], x_0_predict[:, 6:9], x_0_predict[:, 9:12])
                            real_data = iwt(
                                real_data[:, :3], real_data[:, 3:6], real_data[:, 6:9], real_data[:, 9:12])
                            resoluted = iwt(
                                resoluted[:, :3], resoluted[:, 3:6], resoluted[:, 6:9], resoluted[:, 9:12])
                            resoluted_train = iwt(
                                resoluted_train[:, :3], resoluted_train[:, 3:6], resoluted_train[:, 6:9], resoluted_train[:, 9:12])
                        else: 
                            x_0_predict = iwt((x_0_predict[:, :3], [torch.stack(
                                        (x_0_predict[:, 3:6], x_0_predict[:, 6:9], x_0_predict[:, 9:12]), dim=2)]))
                            real_data = iwt((real_data[:, :3], [torch.stack(
                                        (real_data[:, 3:6], real_data[:, 6:9], real_data[:, 9:12]), dim=2)]))
                            resoluted = iwt((resoluted[:, :3], [torch.stack(
                                        (resoluted[:, 3:6], resoluted[:, 6:9], resoluted[:, 9:12]), dim=2)]))
                            resoluted = iwt((resoluted_train[:, :3], [torch.stack(
                                        (resoluted_train[:, 3:6], resoluted_train[:, 6:9], resoluted_train[:, 9:12]), dim=2)]))

                        x_0_predict = (torch.clamp(x_0_predict, -1, 1) + 1) / 2  # 0-1
                        real_data = (torch.clamp(real_data, -1, 1) + 1) / 2  # 0-1
                        resoluted = (torch.clamp(resoluted, -1, 1) + 1) / 2  # 0-1
                        resoluted_train = (torch.clamp(resoluted_train, -1, 1) + 1) / 2  # 0-1

                        # saving predicted samples
                        torchvision.utils.save_image(x_0_predict, os.path.join(
                            exp_path, 'x0_prediction_epoch_{}_iteration_{}.png'.format(epoch,global_step)), normalize=True)

                        #saving real data
                        torchvision.utils.save_image(
                            real_data, os.path.join(exp_path, 'real_data_epoch_{}_iteration_{}.png'.format(epoch,global_step)))

                        # saving resoluted test set images
                        torchvision.utils.save_image(resoluted, os.path.join(
                            exp_path, 'resoluted_test_epoch_{}_iteration_{}.png'.format(epoch,global_step)), normalize=True)

                        # saving resoluted train set images
                        torchvision.utils.save_image(resoluted_train, os.path.join(
                            exp_path, 'resoluted_train_epoch_{}_iteration_{}.png'.format(epoch,global_step)), normalize=True)

        if not args.no_lr_decay:
            schedulerG.step()
            schedulerD.step()




# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser('ddgan parameters')
    parser.add_argument('--seed', type=int, default=42,
                        help='seed used for initialization')

    parser.add_argument('--resume', action='store_true', default=False)

    parser.add_argument('--image_size', type=int, default=32,
                        help='size of image')
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
    parser.add_argument('--attn_resolutions', default=(16,), nargs='+', type=int,
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

    # generator and training
    parser.add_argument(
        '--exp', default='experiment_cifar_default', help='name of experiment')
    parser.add_argument('--dataset', default='cifar10', help='name of dataset')
    parser.add_argument('--datadir', default='./data')
    parser.add_argument('--num_timesteps', type=int, default=4)

    parser.add_argument('--cond_emb_dim', type=int, default=256)
    parser.add_argument('--t_emb_dim', type=int, default=256)
    parser.add_argument('--batch_size', type=int,
                        default=128, help='input batch size')
    parser.add_argument('--num_epoch', type=int, default=1200)
    parser.add_argument('--ngf', type=int, default=64)

    parser.add_argument('--lr_g', type=float,
                        default=1.5e-4, help='learning rate g')
    parser.add_argument('--lr_d', type=float, default=1e-4,
                        help='learning rate d')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.9,
                        help='beta2 for adam')
    parser.add_argument('--no_lr_decay', action='store_true', default=False)

    parser.add_argument('--use_ema', action='store_true', default=False,
                        help='use EMA or not')
    parser.add_argument('--ema_decay', type=float,
                        default=0.9999, help='decay rate for EMA')

    parser.add_argument('--r1_gamma', type=float,
                        default=0.05, help='coef for r1 reg')
    parser.add_argument('--lazy_reg', type=int, default=None,
                        help='lazy regulariation.')

    # wavelet GAN
    parser.add_argument("--current_resolution", type=int, default=256)
    parser.add_argument("--use_pytorch_wavelet", action="store_true")
    parser.add_argument("--rec_loss", action="store_true")
    parser.add_argument("--net_type", default="wavelet")
    parser.add_argument("--num_disc_layers", default=6, type=int)
    parser.add_argument("--no_use_fbn", action="store_true")
    parser.add_argument("--no_use_freq", action="store_true")
    parser.add_argument("--no_use_residual", action="store_true")

    parser.add_argument('--save_content', action='store_true', default=False)
    parser.add_argument('--save_content_every', type=int, default=500,
                        help='save content for resuming every x iterations')
    parser.add_argument('--save_ckpt_every', type=int,default=500, 
                        help='save ckpt every x iterations')

    # ddp
    parser.add_argument('--num_proc_node', type=int, default=1,
                        help='The number of nodes in multi node env.')
    parser.add_argument('--num_process_per_node', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--node_rank', type=int, default=0,
                        help='The index of node.')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='rank of process in the node')
    parser.add_argument('--master_address', type=str, default='127.0.0.1',
                        help='address for master')
    parser.add_argument('--master_port', type=str, default='6002',
                        help='port for master')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='num_workers')

    # super resolution
    parser.add_argument('--l_resolution', type=int, default=16,
                        help='low resolution need to super_resolution')
    parser.add_argument('--h_resolution', type=int, default=128,
                        help='high resolution need to super_resolution')
    

    args = parser.parse_args()

    args.world_size = args.num_proc_node * args.num_process_per_node
    size = args.num_process_per_node

    if size > 1:
        processes = []
        for rank in range(size):
            args.local_rank = rank
            global_rank = rank + args.node_rank * args.num_process_per_node
            global_size = args.num_proc_node * args.num_process_per_node
            args.global_rank = global_rank
            print('Node rank %d, local proc %d, global proc %d' %
                  (args.node_rank, rank, global_rank))
            p = Process(target=init_processes, args=(
                global_rank, global_size, train, args))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        print('starting in debug mode')

        init_processes(0, size, train, args)
