from dataset import *
import os
import torch
from utils import *
import torch.nn as nn
import skimage.io
import argparse
from torchvision import transforms
import tifffile
import pandas as pd
from pytorch_msssim import MS_SSIM
from math import log10, sqrt
import torch.optim.lr_scheduler as lr_scheduler
from model import *
from tensorboardX import SummaryWriter

## setup parse
parser = argparse.ArgumentParser(description='Train the network',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--mode', default='train', choices=['train', 'debug'], dest='mode')
parser.add_argument('--train_continue', default='off',  dest='train_continue')
parser.add_argument('--computer', default='local',choices=['local', 'scc'], dest='computer')
parser.add_argument("--num_gpu", type=int, default=[1], dest='num_gpu')
parser.add_argument('--num_epoch', type=int,  default=150, dest='num_epoch')
parser.add_argument('--batch_size', type=int, default=4, dest='batch_size')
parser.add_argument('--lr', type=float, default=5e-5, dest='lr')
parser.add_argument('--train_ratio', type=float, default=0.9, dest='train_ratio')
parser.add_argument('--dir_chck', default='./checkpoints', dest='dir_chck')
parser.add_argument('--dir_log', default='./log', dest='dir_log')
parser.add_argument('--dir_save', default='./save', dest='dir_save')
parser.add_argument('--num_freq_save', type=int,  default=10, dest='num_freq_save')
parser.add_argument("--local_rank", type=int, default=0, dest='local_rank')
parser.add_argument("--early_stop", type=int, default=50, dest='early_stop', help='cancel=None')
parser.add_argument("--num_psf", type=int, default=9)
parser.add_argument("--network", default='svfourier', help='multiwiener svfourier and cm2net')
parser.add_argument("--ks", type=float, default=10.0)
parser.add_argument("--ps", type=int, default=1)

if __name__ == '__main__':
    PARSER = Parser(parser)
    args = PARSER.get_arguments()
    PARSER.write_args()
    PARSER.print_args()

    torch.manual_seed(3407)
    torch.cuda.empty_cache()
    args.device = torch.device(0)

    if args.computer=='local':
        # change index(starts at 1)
        args.dir_data = 'T:/simulation beads/2d/debug/'
    elif args.computer=='scc':
        args.dir_data='/ad/eng/research/eng_research_cisl/yqw/simulation beads/2d/lsv_2d_dataset/'

    #make dir
    dir_result_val = args.dir_save + '/val/'
    dir_result_train = args.dir_save + '/train/'
    if not os.path.exists(os.path.join(dir_result_train)):
        os.makedirs(os.path.join(dir_result_train))
    if not os.path.exists(os.path.join(dir_result_val)):
        os.makedirs(os.path.join(dir_result_val))

    # training data
    if args.network == 'cm2net':
        # Create the complete dataset
        transform_train = transforms.Compose([Noisecm2(), ToTensorcm2(), Crop()])
        whole_set = CM2Dataset(args.dir_data, transform=transform_train)
        length = len(whole_set)
        train_size, validate_size = int(args.train_ratio * length), length - int(args.train_ratio * length)
        train_set, validate_set = torch.utils.data.random_split(whole_set, [train_size, validate_size])
        train_set = Subset(train_set, isVal=False)
        validate_set = Subset(validate_set, isVal=True)
    else:
        transform_train = transforms.Compose([Noise(), Resize(), ToTensor()])
        whole_set = MyDataset(args.dir_data, transform=transform_train)
        length = len(whole_set)
        train_size, validate_size = int(args.train_ratio*length), length-int(args.train_ratio*length)
        train_set, validate_set = torch.utils.data.random_split(whole_set, [train_size, validate_size])
        print('training images:', len(train_set),
            'testing images:', len(validate_set))

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, num_workers=1, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(validate_set, batch_size=args.batch_size, num_workers=1, shuffle=False, drop_last=True)

    num=len(args.num_gpu)
    num_batch_train = int((train_size / (args.batch_size*num)) + ((train_size % (args.batch_size*num)) != 0))
    num_batch_val = int((validate_size / args.batch_size) + ((validate_size % args.batch_size) != 0))

    ## setup network TBD!
    if args.network == 'multiwiener':
        psfs = skimage.io.imread(args.dir_data + '/psf_v11.tif')
        psfs = np.array(psfs)
        psfs = psfs.astype('float32') / psfs.max()
        psfs = psfs[:,57 * 2:3000, 94 * 2 + 156:4000 - 156]
        psfs = np.pad(psfs, ((0,0),(657, 657), (350, 350)))
        Ks = args.ks*np.ones((args.num_psf, 1, 1))
        deconvolution= MultiWienerDeconvolution2D(psfs,Ks).to(args.device)
        enhancement = RCAN(args.num_psf).to(args.device)
        model = LSVEnsemble2d(deconvolution, enhancement)

    if args.network == 'svfourier':
        deconvolution = FourierDeconvolution2D_ds(args.num_psf,args.ps).to(args.device)
        enhancement = RCAN(args.num_psf).to(args.device)
        model = LSVEnsemble2d(deconvolution, enhancement)

    if args.network == 'cm2net':
        layers = 20  # number of resblocks
        model = cm2net(numBlocks=layers, stackchannels=args.num_psf).to(args.device)  # the input is stack of 9 demixed views, output is one final reconstrution

    #multiple gpu
    model = model.to(args.device)

    ## setup loss & optimization
    ssim_loss = MS_SSIM(data_range=1, size_average=True, channel=1)
    l2_loss = nn.MSELoss()
    bce_loss = nn.BCELoss()
    params = model.parameters()
    optimizer = torch.optim.Adam(params, lr=args.lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, 50, eta_min = 1e-6)


    ## load from checkpoints
    st_epoch = 0

    # Logger
    losslogger = pd.DataFrame()
    if args.train_continue == 'on':
        model, optimizer, st_epoch, losslogger = load(args.dir_chck, model, optimizer, epoch=[], mode=args.mode)

    #save best model
    best_ssim = 0
    trigger = 0
    best_loss=10e7

    ## setup tensorboard
    #set up tensorboard
    dir_log= args.dir_log
    if not os.path.exists(os.path.join(dir_log)):
        os.makedirs(os.path.join(dir_log))
    writer = SummaryWriter(log_dir=dir_log)

    for epoch in range(st_epoch + 1, args.num_epoch + 1):
        ## training phase
        model.train()
        loss_train = []
        ssim_train = []
        psnr_train = []
        for batch, data in enumerate(train_loader, 1):
            def should(freq):
                return freq > 0 and (batch % freq == 0 or batch == num_batch_train)

            # gt shape [Batch,H,W], Output [Batch,1,H,W]
            if args.network == 'cm2net':
                meas = data['meas'].to(args.device)
                gt = data['gt'].to(args.device)
                demix = data['demix'].to(args.device)
                optimizer.zero_grad()
                demix_output, output = model(meas)
                loss_demix = bce_loss(demix_output, demix) + l2_loss(demix_output, demix)
                loss_recon = bce_loss(torch.squeeze(output, 1), gt) + l2_loss(torch.squeeze(output, 1), gt)
                loss = loss_demix+loss_recon
            else:
                meas = data['meas'].to(args.device)
                gt = data['gt'].to(args.device)
                optimizer.zero_grad()
                output = model(meas)
                loss = bce_loss(torch.squeeze(output, 1), gt) + l2_loss(torch.squeeze(output, 1), gt)
            loss.backward()
            optimizer.step()
            output_n = (output - torch.min(output)) / (torch.max(output) - torch.min(output))
            gt_n = (gt - torch.min(gt)) / (torch.max(gt) - torch.min(gt))
            ssim = ssim_loss(output_n, gt_n.unsqueeze(1))
            psnr = 20 * torch.log10(torch.max(output) / sqrt(l2_loss(torch.squeeze(output, 1), gt)))
            # get losses
            loss_train += [loss.item()]
            ssim_train += [ssim.item()]
            psnr_train += [psnr.item()]

            if args.local_rank == 0:
                print('TRAIN: EPOCH %d: BATCH %04d/%04d: LOSS: %.4f SSIM: %.4f'
                      % (epoch, batch, num_batch_train, np.mean(loss_train), np.mean(ssim_train)))

        scheduler.step()

        if args.local_rank == 0 and (epoch % args.num_freq_save) == 0:
            gt = gt.data.cpu().numpy()
            x_recon = torch.squeeze(output,1).data.cpu().numpy()
            for j in range(gt.shape[0]):
                im_gt = (np.clip(gt[j, ...]/ np.max(gt[j, ...]), 0, 1) * 255).astype(np.uint8)
                im_recon = (np.clip(x_recon[j, ...] / np.max(x_recon[j, ...]), 0, 1) * 255).astype(np.uint8)
                tifffile.imwrite((dir_result_train + str(epoch) + '_recon'  + '.tif'),im_recon.squeeze())
                tifffile.imwrite((dir_result_train + str(epoch) + '_gt' +  '.tif'),im_gt.squeeze())

        ## validation phase
        with torch.no_grad():
            model.eval()
            loss_val = []
            ssim_val = []
            psnr_val = []

            for batch, data in enumerate(val_loader, 1):
                # forward simulation(add noise)
                if args.network == 'cm2net':
                  meas = data['meas'].to(args.device)
                  gt = data['gt'].to(args.device)
                  demix = data['demix'].to(args.device)
                  demix_output, output = model(meas)
                  # print(demix_output.shape,demix.shape)
                  loss_demix = bce_loss(demix_output, demix) + l2_loss(demix_output, demix)
                  loss_recon = bce_loss(torch.squeeze(output, 1), gt) + l2_loss(torch.squeeze(output, 1), gt)
                  loss = loss_demix+loss_recon
                else:
                  meas = data['meas'].to(args.device)
                  gt = data['gt'].to(args.device)
                  output = model(meas)
                  loss = bce_loss(torch.squeeze(output, 1), gt)+l2_loss(torch.squeeze(output, 1), gt)
                output_n = (output - torch.min(output)) / (torch.max(output) - torch.min(output))
                gt_n = (gt - torch.min(gt)) / (torch.max(gt) - torch.min(gt))
                ssim = ssim_loss(output_n, gt_n.unsqueeze(1))
                psnr = 20 * torch.log10(torch.max(output) / sqrt(l2_loss(torch.squeeze(output, 1), gt)))
                # get losses
                loss_val += [loss.item()]
                ssim_val += [ssim.item()]
                psnr_val += [psnr.item()]

                if args.local_rank == 0:
                    print('VALID: EPOCH %d: BATCH %04d/%04d: LOSS: %.4f SSIM: %.4f'
                          % (epoch, batch, num_batch_val, np.mean(loss_val), np.mean(ssim_val)))

            if epoch == 1:
                gt = gt.data.cpu().numpy()
                im_gt = (np.clip(gt[-1, ...] / np.max(gt[-1, ...]), 0, 1) * 255).astype(np.uint8)
                tifffile.imwrite((dir_result_val + str(epoch) + '_gt' + '.tif'), im_gt.squeeze())

            if args.local_rank == 0 and (epoch % args.num_freq_save) == 0:
                x_recon = output.data.cpu().numpy()
                im_recon = (np.clip(x_recon[-1, ...] / np.max(x_recon[-1, ...]), 0, 1) * 255).astype(np.uint8)
                tifffile.imwrite((dir_result_val + str(epoch) + '_recon' + '.tif'), im_recon.squeeze())

                if args.network == 'svfourier':
                    psfs_re = model.deconvolution.psfs_re.detach().cpu().numpy()
                    psfs_im = model.deconvolution.psfs_im.detach().cpu().numpy()
                    psf_freq = psfs_re + psfs_im * 1j
                    psf = np.fft.ifftshift(np.fft.irfft2(psf_freq, axes=(-2, -1)))
                    psf_mip = np.max(psf, 0).squeeze()
                    psf_mip = (psf_mip / np.abs(psf_mip).max() * 65535.0).astype('int16')
                    tifffile.imwrite((dir_result_val + str(epoch) + '_psf_mip' + '.tif'), psf_mip, photometric='minisblack')

                if args.network == 'multiwiener':
                    psf = model.deconvolution.psfs.detach().cpu().numpy()
                    psf_mip = np.max(psf, 0).squeeze()
                    psf_mip = (psf_mip / np.abs(psf_mip).max() * 65535.0).astype('int16')
                    tifffile.imwrite((dir_result_val + str(epoch) + '_psf_mip' + '.tif'), psf_mip, photometric='minisblack')

        if args.local_rank == 0:
            # set in logs
            df = pd.DataFrame()
            df['loss_train'] = pd.Series(np.mean(loss_train))
            df['ssim_train'] = pd.Series(np.mean(ssim_train))
            df['psnr_train'] = pd.Series(np.mean(psnr_train))
            df['loss_val'] = pd.Series(np.mean(loss_val))
            df['ssim_val'] = pd.Series(np.mean(ssim_val))
            df['psnr_val'] = pd.Series(np.mean(psnr_val))
            losslogger = losslogger.append(df)
            writer.add_scalar('Loss/loss_train', np.mean(loss_train), epoch)
            writer.add_scalar('SSIM/ssim_train', np.mean(ssim_train), epoch)
            writer.add_scalar('PSNR/psnr_train', np.mean(psnr_train), epoch)
            writer.add_scalar('Loss/loss_val', np.mean(loss_val), epoch)
            writer.add_scalar('SSIM/ssim_val', np.mean(ssim_val), epoch)
            writer.add_scalar('PSNR/psnr_val', np.mean(psnr_val), epoch)

        trigger += 1
        if args.local_rank == 0 and (np.mean(ssim_val) > best_ssim):
            save(args.dir_chck+ '/best_model/', model, optimizer, epoch, losslogger)
            best_ssim = np.mean(ssim_val)
            print("=>saved best model")
            trigger = 0

        if not args.early_stop is not None and args.local_rank == 0:
            if trigger >= args.early_stop:
                print("=> early stop")
            break

        # save checkpoint
        if args.local_rank == 0 and (epoch % args.num_freq_save) == 0:
            save(args.dir_chck, model, optimizer,epoch,losslogger)
