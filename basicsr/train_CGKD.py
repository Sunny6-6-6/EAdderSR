import argparse
import datetime
import logging
import math
import random
import time
from os import path as osp
import sys
import os
from collections import OrderedDict
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__,),),'..'))

from basicsr.models.archs.KD import P_KD
import torch
from basicsr.data import create_dataloader, create_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.models import create_model
from basicsr.utils import (MessageLogger, check_resume, get_env_info,
                           get_root_logger, get_time_str, init_tb_logger,
                           init_wandb_logger, make_exp_dirs, mkdir_and_rename,
                           set_random_seed)
from basicsr.utils.dist_util import get_dist_info, init_dist
from basicsr.utils.options import dict2str, parse


def parse_options(is_train=True):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-opt', type=str, required=True, help='Path to option YAML file.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = parse(args.opt, is_train=is_train)

    # distributed settings
    if args.launcher == 'none':
        opt['dist'] = False
        print('Disable distributed.', flush=True)
    else:
        opt['dist'] = True
        if args.launcher == 'slurm' and 'dist_params' in opt:
            init_dist(args.launcher, **opt['dist_params'])
        else:
            init_dist(args.launcher)

    opt['rank'], opt['world_size'] = get_dist_info()

    # random seed
    seed = opt.get('manual_seed')
    if seed is None:
        seed = random.randint(1, 10000)
        opt['manual_seed'] = seed
    set_random_seed(seed + opt['rank'])

    return opt


def init_loggers(opt):
    log_file = osp.join(opt['path']['log'],
                        f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(
        logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # initialize wandb logger before tensorboard logger to allow proper sync:
    if (opt['logger'].get('wandb')
            is not None) and (opt['logger']['wandb'].get('project')
                              is not None) and ('debug' not in opt['name']):
        assert opt['logger'].get('use_tb_logger') is True, (
            'should turn on tensorboard when using wandb')
        init_wandb_logger(opt)
    tb_logger = None
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
        tb_logger = init_tb_logger(log_dir=osp.join('tb_logger', opt['name']))
    return logger, tb_logger


def create_train_val_dataloader(opt, logger):
    # create train and val dataloaders
    train_loader, val_loader = None, None
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            train_set = create_dataset(dataset_opt)
            train_sampler = EnlargedSampler(train_set, opt['world_size'],
                                            opt['rank'], dataset_enlarge_ratio)
            train_loader = create_dataloader(
                train_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=train_sampler,
                seed=opt['manual_seed'])

            num_iter_per_epoch = math.ceil(
                len(train_set) * dataset_enlarge_ratio /
                (dataset_opt['batch_size_per_gpu'] * opt['world_size']))
            total_iters = int(opt['train']['total_iter'])
            total_epochs = math.ceil(total_iters / (num_iter_per_epoch))
            logger.info(
                'Training statistics:'
                f'\n\tNumber of train images: {len(train_set)}'
                f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
                f'\n\tBatch size per gpu: {dataset_opt["batch_size_per_gpu"]}'
                f'\n\tWorld size (gpu number): {opt["world_size"]}'
                f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
                f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')

        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(
                val_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=None,
                seed=opt['manual_seed'])
            logger.info(
                f'Number of val images/folders in {dataset_opt["name"]}: '
                f'{len(val_set)}')
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')

    return train_loader, train_sampler, val_loader, total_epochs, total_iters


def CGKD_optimizer(teacher, student, pkd, pkd_optim):

################ ANN ################
    l_total_a = 0
    loss_dict_a = OrderedDict()
    # pixel loss

    pkd.hooks.reset()
    teacher.output = teacher.net_g(teacher.lq)
    student.output = student.net_g(student.lq)

    if student.cri_pix:
        # pass
        l_pix_a = student.cri_pix(student.output, student.gt)
        l_total_a += l_pix_a
        loss_dict_a['l_pix'] = l_pix_a

    if student.cri_perceptual:
        l_percep, l_style = student.cri_perceptual(student.output, student.gt)
        if l_percep is not None:
            l_total_a += l_percep * 0.01
            loss_dict_a['l_percep'] = l_percep * 0.01
        if l_style is not None:
            l_total_a += l_style
            loss_dict_a['l_style'] = l_style

    ################ gan loss (if needed)

    # print(l_total_c)
    pkd_loss = pkd.forward()
    loss_dict_a['l_pkd'] = pkd_loss

    l_total_a = l_total_a + pkd_loss
    loss_dict_a['l_total'] = l_total_a

    student.optimizer_g.zero_grad()
    l_total_a.backward()
    student.optimizer_g.step()

    ################# optimize net_d
    for p in student.net_d.parameters():
        p.requires_grad = True

    student.optimizer_d.zero_grad()
    # real
    real_d_pred = student.net_d(student.gt)
    l_d_real = student.cri_gan(real_d_pred, True, is_disc=True)
    loss_dict_a['l_d_real'] = l_d_real
    loss_dict_a['out_d_real'] = torch.mean(real_d_pred.detach())
    l_d_real.backward()
    
    # fake
    fake_d_pred = student.net_d(student.output.detach())
    l_d_fake = student.cri_gan(fake_d_pred, False, is_disc=True)
    loss_dict_a['l_d_fake'] = l_d_fake
    loss_dict_a['out_d_fake'] = torch.mean(fake_d_pred.detach())
    l_d_fake.backward()
    student.optimizer_d.step()
    
    student.log_dict = student.reduce_loss_dict(loss_dict_a)

def main():
    # parse options, set distributed setting, set random seed
    opt = parse_options(is_train=True)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # load resume states if necessary
    if opt['path'].get('resume_state'):
        device_id = torch.cuda.current_device()
        resume_state = torch.load(
            opt['path']['resume_state'],
            map_location=lambda storage, loc: storage.cuda(device_id))
    else:
        resume_state = None

    # mkdir for experiments and logger
    if resume_state is None:
        make_exp_dirs(opt)
        if opt['logger'].get('use_tb_logger') and 'debug' not in opt[
                'name'] and opt['rank'] == 0:
            mkdir_and_rename(osp.join('tb_logger', opt['name']))

    # initialize loggers
    logger, tb_logger = init_loggers(opt)

    # create train and validation dataloaders
    result = create_train_val_dataloader(opt, logger)
    train_loader, train_sampler, val_loader, total_epochs, total_iters = result

    # create model
    if resume_state:  # resume training
        opt_ = opt.copy()
        check_resume(opt, resume_state['iter'])
        opt_['network_g']['type'] = 'MSRResNet_Adder'
        student = create_model(opt_)
        student.resume_training(resume_state)  # handle optimizers and schedulers
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, "
                    f"iter: {resume_state['iter']}.")

        opt['network_g']['type'] = 'MSRResNet'
        opt['path']['pretrain_network_g'] ='experiments/MSRResNet_cnn_x4/models/net_g_latest.pth'
        teacher = create_model(opt)

        pkd = P_KD(teacher.net_g, student.net_g).cuda()
        pkd_optim = torch.optim.Adam(pkd.parameters(), lr=0.001)
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']

    else:
        opt_ = opt.copy()
        # student
        opt_['network_g']['type'] = 'MSRResNet_Adder'
        student = create_model(opt_)
        # teacher
        opt['network_g']['type'] = 'MSRResNet'
        opt['path']['pretrain_network_g'] ='experiments/MSRResNet_cnn_x4/models/net_g_latest.pth'
        teacher = create_model(opt)

        pkd = P_KD(teacher.net_g, student.net_g).cuda()
        pkd_optim = torch.optim.Adam(pkd.parameters(),lr=0.001)
        start_epoch = 0
        current_iter = 0

    # create message logger (formatted outputs)
    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    # dataloader prefetcher
    prefetch_mode = opt['datasets']['train'].get('prefetch_mode')
    if prefetch_mode is None or prefetch_mode == 'cpu':
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == 'cuda':
        prefetcher = CUDAPrefetcher(train_loader, opt)
        logger.info(f'Use {prefetch_mode} prefetch dataloader')
        if opt['datasets']['train'].get('pin_memory') is not True:
            raise ValueError('Please set pin_memory=True for CUDAPrefetcher.')
    else:
        raise ValueError(f'Wrong prefetch_mode {prefetch_mode}.'
                         "Supported ones are: None, 'cuda', 'cpu'.")

    # training
    logger.info(
        f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    data_time, iter_time = time.time(), time.time()
    start_time = time.time()

    for epoch in range(start_epoch, total_epochs + 1):
        train_sampler.set_epoch(epoch)
        prefetcher.reset()
        train_data = prefetcher.next()

        while train_data is not None:
            data_time = time.time() - data_time

            current_iter += 1
            if current_iter > total_iters:
                break

            # update learning rate
            teacher.update_learning_rate(
                current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))

            student.update_learning_rate(
                current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))

            # training

            teacher.feed_data(train_data)
            student.feed_data(train_data)
            ################### optimize parameters ################

            CGKD_optimizer(teacher,student,pkd,pkd_optim)

            iter_time = time.time() - iter_time
            # log
            if current_iter % opt['logger']['print_freq'] == 0:
                log_vars = {'epoch': epoch, 'iter': current_iter}
                log_vars.update({'lrs': student.get_current_learning_rate()})
                log_vars.update({'time': iter_time, 'data_time': data_time})
                log_vars.update(student.get_current_log())
                msg_logger(log_vars)

            # save models and training states
            if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                student.save(epoch, current_iter)

            # validation
            if opt.get('val') is not None and (current_iter %
                                               opt['val']['val_freq'] == 0):
                teacher.validation(val_loader, current_iter, tb_logger,
                                 opt['val']['save_img'])
                student.validation(val_loader, current_iter, tb_logger,
                                 opt['val']['save_img'])
                pkd.hooks.reset()

            data_time = time.time()
            iter_time = time.time()
            train_data = prefetcher.next()

        # end of iter

    # end of epoch

    consumed_time = str(
        datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'End of training. Time consumed: {consumed_time}')
    logger.info('Save the latest model.')
    model.save(epoch=-1, current_iter=-1)  # -1 stands for the latest
    if opt.get('val') is not None:
        model.validation(val_loader, current_iter, tb_logger,
                         opt['val']['save_img'])
    if tb_logger:
        tb_logger.close()


if __name__ == '__main__':
    main()
