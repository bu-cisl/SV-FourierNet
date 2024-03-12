from __future__ import absolute_import, division, print_function
import torch.distributed as dist
import os
import logging
from math import log10, sqrt
import torch
import torch.optim.lr_scheduler as lr_scheduler
# import argparse

''''
class Logger:
class Parser:
'''
class Parser:
    def __init__(self, parser):
        self.__parser = parser
        self.__args = parser.parse_args()

    def get_parser(self):
        return self.__parser

    def get_arguments(self):
        return self.__args

    def write_args(self):
        params_dict = vars(self.__args)

        log_dir = os.path.join(params_dict['dir_log'])
        args_name = os.path.join(log_dir, 'args.txt')

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        with open(args_name, 'wt') as args_fid:
            args_fid.write('----' * 10 + '\n')
            args_fid.write('{0:^40}'.format('PARAMETER TABLES') + '\n')
            args_fid.write('----' * 10 + '\n')
            for k, v in sorted(params_dict.items()):
                args_fid.write('{}'.format(str(k)) + ' : ' + ('{0:>%d}' % (35 - len(str(k)))).format(str(v)) + '\n')
            args_fid.write('----' * 10 + '\n')

    def print_args(self, name='PARAMETER TABLES'):
        params_dict = vars(self.__args)

        print('----' * 10)
        print('{0:^40}'.format(name))
        print('----' * 10)
        for k, v in sorted(params_dict.items()):
            if '__' not in str(k):
                print('{}'.format(str(k)) + ' : ' + ('{0:>%d}' % (35 - len(str(k)))).format(str(v)))
        print('----' * 10)


class Logger:
    def __init__(self, info=logging.INFO, name=__name__):
        logger = logging.getLogger(name)
        logger.setLevel(info)

        self.__logger = logger

    def get_logger(self, handler_type='stream_handler'):
        if handler_type == 'stream_handler':
            handler = logging.StreamHandler()
            log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(log_format)
        else:
            handler = logging.FileHandler('utils.log')

        self.__logger.addHandler(handler)

        return self.__logger

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def save(dir_chck, model, optimizer, epoch, losslogger):
    if not os.path.exists(dir_chck):
        os.makedirs(dir_chck)

    torch.save({'model': model.state_dict(),
                'optim': optimizer.state_dict(),
                'losslogger': losslogger},
               '%s/model_epoch%04d.pth' % (dir_chck, epoch))

def load(dir_chck, model, optimizer=[], epoch=[], mode='train'):

    if not os.path.exists(dir_chck) or not os.listdir(dir_chck):
        epoch = 0
        if mode == 'train':
            return model, optimizer, epoch
        elif mode == 'test':
            return model, epoch

    if not epoch:
        ckpt = os.listdir(dir_chck)
        ckpt.sort()
        epoch = int(ckpt[-1].split('epoch')[1].split('.pth')[0])
        dict_net = torch.load('%s/model_epoch%04d.pth' % (dir_chck, epoch))
 
    print('Loaded %dth network' % epoch)
    
    if mode == 'train':
        model.load_state_dict(dict_net['model'])
        optimizer.load_state_dict(dict_net['optim'])
        losslogger = dict_net['losslogger']

        return model, optimizer, epoch, losslogger

    elif mode == 'test':
        model.load_state_dict(dict_net['model'])
        losslogger = dict_net['losslogger']

        return model, losslogger

