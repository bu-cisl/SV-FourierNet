"""
@author: yangqianwan
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils import data
import torch.nn
import skimage.io
import glob

class MyDataset(data.Dataset):
    def __init__(self, dir_data,transform=None):
        self.dir_data = dir_data
        self.transform = transform
    def __getitem__(self,index):
        meas = skimage.io.imread(self.dir_data + '/meas_{:n}.tif'.format(index + 1))
        gt = skimage.io.imread(self.dir_data + '/gt_{:n}.tif'.format(index + 1))
        meas = meas[57 * 2:3000, 94 * 2 + 156:4000 - 156]
        data = {'gt': gt.astype('float32') / gt.max(), 'meas': meas.astype('float32') / meas.max()}
        if self.transform is not None:
            data = self.transform(data)
        return data
    def __len__(self):
        return len(glob.glob(self.dir_data + '/meas_*.tif'))

class CM2Dataset(data.Dataset):
    def __init__(self, dir_data,transform=None):
        self.dir_data = dir_data
        self.transform = transform
    def __getitem__(self,index):
        meas = skimage.io.imread(self.dir_data + '/meas_{:n}.tif'.format(index + 1))
        gt = skimage.io.imread(self.dir_data + '/gt_{:n}.tif'.format(index + 1))
        demix = skimage.io.imread(self.dir_data + '/demix_{:n}.tif'.format(index + 1))
        # print(meas.shape,gt.shape,demix.shape)
        data = {'gt': gt.astype('float32') / gt.max(), 'meas': meas.astype('float32') / meas.max(), 'demix': demix.astype('float32') / demix.max()}
        if self.transform is not None:
            data = self.transform(data)
        return data
    def __len__(self):
        return len(glob.glob(self.dir_data + '/meas_*.tif'))


class Subset(data.Dataset):
    # is this pulling all images or just a single image given by idx?
    def __init__(self, dataset, isVal):
        self.dataset = dataset
        self.isVal = isVal

    def __getitem__(self, idx):
        p = 256  # 256x256 random patch size
        if self.isVal:  # return self.dataset.__getitem__(idx)
            data = self.dataset.__getitem__(idx)
            return data
        else:
            data = self.dataset.__getitem__(idx)
            gt, meas, demix = data['gt'], data['meas'], data['demix']
            dim =meas.shape[-1]
            # print(len(self.dataset.__getitem__(idx))) # returns 3
            a = torch.randint(0, dim- p, (1,))  # dim[0] == 2400 (tried dim[0] but that should be =9) #tuple with only one item
            b = torch.randint(0, dim - p, (1,))  # dim[1] == 2400
            data = {'gt': gt[...,a:a + p, b:b + p], 'meas': meas[..., a:a + p, b:b + p], 'demix': demix[..., a:a + p, b:b + p]}
            return data

    def __len__(self):
        return self.dataset.__len__()

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, data):
        # Swap color axis because numpy image: H x W x C
        #                         torch image: C x H x W

        # for key, value in data:
        #     data[key] = torch.from_numpy(value.transpose((2, 0, 1)))
        #
        # return data
        gt, meas = data['gt'], data['meas']
        return {'gt': torch.from_numpy(gt),
                'meas': torch.from_numpy(meas)}

class Noise(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, data):
        gt, meas = data['gt'], data['meas']
        amin = 7.8109e-5
        amax = 9.6636e-5
        bmin = 1.3836e-8
        bmax = 9.6505e-7
        a = np.random.rand(1) * (amax - amin) + amin  # from calibration
        b = np.random.rand(1) * (bmax - bmin) + bmin  # from calibration
        meas += np.sqrt(a * meas + b) * np.random.randn(meas.shape[0], meas.shape[1])
        data = {'gt': gt, 'meas': meas}

        return data

class Resize(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, data):
        gt, meas = data['gt'], data['meas']
        meas = np.pad(meas, ((657, 657), (350, 350)))
        data = {'gt': gt, 'meas': meas}
        return data

class ToTensorcm2(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, data):
        # Swap color axis because numpy image: H x W x C
        #                         torch image: C x H x W

        # for key, value in data:
        #     data[key] = torch.from_numpy(value.transpose((2, 0, 1)))
        #
        # return data
        gt, meas, demix = data['gt'], data['meas'], data['demix']
        return {'gt': torch.from_numpy(gt),
                'meas': torch.from_numpy(meas),
                'demix': torch.from_numpy(demix)}

class Noisecm2(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, data):
        gt, meas, demix = data['gt'], data['meas'], data['demix']
        amin = 7.8109e-5
        amax = 9.6636e-5
        bmin = 1.3836e-8
        bmax = 9.6505e-7
        a = np.random.rand(1) * (amax - amin) + amin  # from calibration
        b = np.random.rand(1) * (bmax - bmin) + bmin  # from calibration
        meas += np.sqrt(a * meas + b) * np.random.randn(meas.shape[0], meas.shape[1])
        data = {'gt': gt, 'meas': meas, 'demix': demix}

        return data
class Crop(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, data):
        gt, meas, demix = data['gt'], data['meas'], data['demix']
        tot_len= 2400
        tmp_pad = 900
        meas = F.pad(meas, (tmp_pad, tmp_pad, tmp_pad, tmp_pad), 'constant', 0)

        loc = [(664, 1192), (664, 2089), (660, 2982),
               (1564, 1200), (1557, 2094), (1548, 2988),
               (2460, 1206), (2452, 2102), (2444, 2996)]

        meas = torch.stack([
            meas[x - (tot_len // 2) + tmp_pad:x + (tot_len // 2) + tmp_pad,
            y - (tot_len // 2) + tmp_pad:y + (tot_len // 2) + tmp_pad] for x, y in loc
        ])
        # print(meas.shape,gt.shape,demix.shape)
        data = {'gt': gt, 'meas': meas, 'demix':demix}
        return data
