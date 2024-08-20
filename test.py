import skimage.io
import argparse
import tifffile
from model import *

## setup parse
parser = argparse.ArgumentParser(description='Train the network', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dir_data', default='test/', dest='dir_data')
parser.add_argument("--network", default='multiwiener', help='multiwiener svfourier and cm2net')
parser.add_argument('--model_name', default='/multiwiener', dest='model_name')
parser.add_argument('--batch_size', type=int, default=1, dest='batch_size')
parser.add_argument("--local_rank", type=int, default=0, dest='local_rank')
parser.add_argument("--num_psf", type=int, default=9)
parser.add_argument("--ps", type=int, default=1)
parser.add_argument("--ks", type=float, default=10.0)
parser.add_argument("--epoch", type=int, default=54)
parser.add_argument('--dir_chck', default='./checkpoints', dest='dir_chck')
parser.add_argument("--distributed", type=bool, default=False, dest='distributed')
parser.add_argument('--lr', type=float, default=5e-5, dest='lr')
parser.add_argument('--mode', default='test', choices=['train', 'test'], dest='mode')

args = parser.parse_args(''.split())

# make dir
dir_result_test = args.dir_data + args.model_name
if not os.path.exists(os.path.join(dir_result_test)):
    os.makedirs(os.path.join(dir_result_test))

args.num_gpu = list(range(torch.cuda.device_count()))
torch.cuda.set_device(args.local_rank)
args.device=torch.device(f'cuda:{args.local_rank}')

if args.network == 'multiwiener':
    psfs = skimage.io.imread('pretrained_model\psf.tif')
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

params = model.parameters()
dict_net = torch.load('pretrained_model/%s.pth' % (args.model_name))
model.load_state_dict(dict_net)
print('Successfully loaded network %s' % (args.network))

with torch.no_grad():
    model.eval()
    meas = skimage.io.imread(args.dir_data + 'meas.tif').astype('float32')
    if args.network == 'cm2net':
        tot_len = 2400
        tmp_pad = 900
        meas = np.pad(meas, ((tmp_pad, tmp_pad), (tmp_pad, tmp_pad)), 'constant')
        loc = [(664, 1192), (664, 2089), (660, 2982),
               (1564, 1200), (1557, 2094), (1548, 2988),
               (2460, 1206), (2452, 2102), (2444, 2996)]
        meas = np.stack([
            meas[x - (tot_len // 2) + tmp_pad:x + (tot_len // 2) + tmp_pad,
            y - (tot_len // 2) + tmp_pad:y + (tot_len // 2) + tmp_pad] for x, y in loc
        ])
    else:
        meas = meas[57 * 2:3000, 94 * 2 + 156:4000 - 156]
        meas = np.pad(meas, ((657, 657), (350, 350)))

    meas = torch.from_numpy(meas / meas.max()).unsqueeze(0)
    meas = meas.to(args.device)
    if args.network == 'cm2net':
        demix_output, output = model(meas)
    else:
        output = model(meas)

    output_n = (output - torch.min(output)) / (torch.max(output) - torch.min(output))
    x_recon = output.data.cpu().numpy().squeeze()
    im_recon = (np.clip(x_recon / np.max(x_recon), 0, 1) * 65535).astype(np.uint16)
    tifffile.imwrite((dir_result_test + '/recon.tif'), im_recon.squeeze())



