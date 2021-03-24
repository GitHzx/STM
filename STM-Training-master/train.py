from libs.dataset.data import ROOT, DATA_CONTAINER, multibatch_collate_fn
from libs.dataset.transform import TrainTransform, TestTransform
from libs.utils.logger import Logger, AverageMeter
from libs.utils.loss import *
from libs.utils.utility import write_mask, save_checkpoint, adjust_learning_rate
from libs.models.models import STAN

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

import numpy as np
import os
import os.path as osp
import shutil
import time
import pickle
import argparse
from progress.bar import Bar
from collections import OrderedDict

from options import OPTION as opt

MAX_FLT = 1e6

''' 命令行参数接口函数 '''
# 仅通过命令行控制 GPU 参数，其他参数在 option.py 中设置
def parse_args():
    parser = argparse.ArgumentParser('Training Mask Segmentation') # 创建解析器，用于将命令行解析成 Python 数据类型
    parser.add_argument('--gpu', default='', type=str, help='set gpu id to train the network') # 添加参数
    return parser.parse_args() # 解析参数并返回

''' 主函数 '''
def main():

    start_epoch = 0

    args = parse_args() # 接收解析后的参数
    
    ''' Use GPU '''
    # 处理当前系统的信息，信息键值对的形式存储
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu if args.gpu != '' else str(opt.gpu_id) # 若未输入 GPU 参数，则使用 option.py 中定义默认参数
    use_gpu = torch.cuda.is_available() and (args.gpu != '' or int(opt.gpu_id)) >= 0 # 使用 GPU 的条件：cuda可用，GPU 参数不为空

    ''' Make checkpoint direction '''
    # 创建模型训练存储检查点路径
    if not os.path.isdir(opt.checkpoint):
        os.makedirs(opt.checkpoint)

    ''' Preprocess data '''
    print('==> Preparing dataset')
    input_dim = opt.input_size # 设置图像输入尺寸
    # 数据预处理及数据增强处理
    train_transformer = TrainTransform(size=input_dim) # 训练数据处理中包含了多种增强策略
    test_transformer = TestTransform(size=input_dim)

    ''' Construct train data '''
    try:
        # isinstance() 判断一个对象的类型，考虑继承关系
        # zip() 将可迭代的对象中对应的元素打包成元组，然后返回由这些元组组成的列表

        # 若 opt.trainset 为列表，则训练集包括多个不同的数据集，需要分别按照不同数据集要求进行处理
        if isinstance(opt.trainset, list): 
            datalist = [] 
            # 不同的数据集使用不同的帧率与 max_skip，将数据集类别与其 datafreq 及 max_skip 对应打包
            for dataset, freq, max_skip in zip(opt.trainset, opt.datafreq, opt.max_skip):
                # 按照给定的数据格式对数据元组进行处理，返回 frame, mask, num_obj, info
                # 数据格式处理类 DATA_CONTAINER 在 data.py 中定义
                ds = DATA_CONTAINER[dataset](
                    train=True, 
                    sampled_frames=opt.sampled_frames, 
                    transform=train_transformer, 
                    max_skip=max_skip, 
                    samples_per_video=opt.samples_per_video
                )
                datalist += [ds] * freq # 将 ds 转成列表元素复制 freq 次，依次存储进 datalist，例 x += [1] * 3，则 x 为 [[1],[1],[1]]

            # torch.utils.data.ConcatDataset() 用于将包含多个子数据集的列表打包成一整个数据集
            trainset = data.ConcatDataset(datalist)
        # 若 opt.trainset 不是列表，则训练集只包含一个数据集
        else:
            # 设置训练的 max skip time length 
            max_skip = opt.max_skip[0] if isinstance(opt.max_skip, list) else opt.max_skip
            trainset = DATA_CONTAINER[opt.trainset](
                train=True, 
                sampled_frames=opt.sampled_frames, 
                transform=train_transformer, 
                max_skip=max_skip, 
                samples_per_video=opt.samples_per_video
                )
    except KeyError as ke:
        print('[ERROR] invalide dataset name is encountered. The current acceptable datasets are:')
        print(list(DATA_CONTAINER.keys()))
        exit()

    ''' Construct test data '''
    testset = DATA_CONTAINER[opt.valset](
        train=False,
        transform=test_transformer,
        samples_per_video=1
        )

    ''' Load data '''
    trainloader = data.DataLoader(trainset, batch_size=opt.train_batch, shuffle=True, num_workers=opt.workers,
                                  collate_fn=multibatch_collate_fn)

    testloader = data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=opt.workers,
                                 collate_fn=multibatch_collate_fn)
    ''' Model '''
    print("==> creating model")

    net = STAN(opt.keydim, opt.valdim) # 模型实例，opt.keydim, opt.valdim 分别代表 key 维度与 value 维度
    print('Total params: %.2fM' % (sum(p.numel() for p in net.parameters())/1000000.0))

    ''' Train/Test '''
    net.eval() # 将实例化的 model 指定 train/eval 分别进行训练和测试，eval()会固定 BN 和 DropOut

    ''' Use GPU '''
    if use_gpu:
        net = net.cuda() # cuda()就将模型转换为GPU的类型

    # set training parameters
    for p in net.parameters():
        p.requires_grad = True # 保留对应参数的梯度信息进行梯度下降

    ''' Choose loss function '''
    criterion = None
    celoss = cross_entropy_loss
    if opt.loss == 'ce':
        criterion = celoss
    elif opt.loss == 'iou':
        criterion = mask_iou_loss
    elif opt.loss == 'both':
        criterion = lambda pred, target, obj: celoss(pred, target, obj) + mask_iou_loss(pred, target, obj)
    else:
        raise TypeError('unknown training loss %s' % opt.loss)

    ''' Choose optimizer '''
    optimizer = None
    if opt.solver == 'sgd':

        optimizer = optim.SGD(net.parameters(), lr=opt.learning_rate,
                        momentum=opt.momentum[0], weight_decay=opt.weight_decay)
    elif opt.solver == 'adam':

        optimizer = optim.Adam(net.parameters(), lr=opt.learning_rate,
                        betas=opt.momentum, weight_decay=opt.weight_decay)
    else:
        raise TypeError('unkown solver type %s' % opt.solver)

    ''' Resume from checkpoint '''
    title = 'STAN'
    minloss = float('inf')
    # 若路径不存在则建立路径
    opt.checkpoint = osp.join(osp.join(opt.checkpoint, opt.valset))
    if not osp.exists(opt.checkpoint):
        os.mkdir(opt.checkpoint)
    # 通过参数 opt.resume 控制 Resume
    if opt.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint {}'.format(opt.resume))
        assert os.path.isfile(opt.resume), 'Error: no checkpoint directory found!'
        # opt.checkpoint = os.path.dirname(opt.resume)
        checkpoint = torch.load(opt.resume)
        minloss = checkpoint['minloss']
        start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        skips = checkpoint['max_skip']

    # set_max_skip？？？？？？？？？两个？？？？？？？？？？
        try:
            if isinstance(skips, list):
                for idx, skip in enumerate(skips):
                    trainloader.dataset.datasets[idx].set_max_skip(skip)
            else:
                trainloader.dataset.set_max_skip(skip)
        except:
            print('[Warning] Initializing max skip fail')

        logger = Logger(os.path.join(opt.checkpoint, opt.mode+'_log.txt'), resume=True) # 日志记录
    else:
        if opt.initial:
            print('==> Initialize model with weight file {}'.format(opt.initial))
            weight = torch.load(opt.initial)
            if isinstance(weight, OrderedDict):
                net.load_param(weight)
            else:
                net.load_param(weight['state_dict'])
        
        logger = Logger(os.path.join(opt.checkpoint, opt.mode+'_log.txt'), resume=False) # 日志记录
        start_epoch = 0

    logger.set_items(['Epoch', 'LR', 'Train Loss'])

    ''' Train and val '''
    # 调节学习率？？？？？？？？？两个？？？？？？？？？？
    for epoch in range(start_epoch):
        adjust_learning_rate(optimizer, epoch, opt)
    # 进行训练或测试
    for epoch in range(start_epoch, opt.epochs):

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, opt.epochs, opt.learning_rate))
        adjust_learning_rate(optimizer, epoch, opt)

        train_loss = train(trainloader,
                           model=net,
                           criterion=criterion,
                           optimizer=optimizer,
                           epoch=epoch,
                           use_cuda=use_gpu,
                           iter_size=opt.iter_size,
                           mode=opt.mode,
                           threshold=opt.iou_threshold)

        if (epoch + 1) % opt.epoch_per_test == 0:
            test_loss = test(testloader,
                            model=net,
                            criterion=criterion,
                            epoch=epoch,
                            use_cuda=use_gpu,
                            opt=opt)

        # append logger file
        logger.log(epoch+1, opt.learning_rate, train_loss)

        # adjust max skip
        if (epoch + 1) % opt.epochs_per_increment == 0:
            if isinstance(trainloader.dataset, data.ConcatDataset):
                for dataset in trainloader.dataset.datasets:
                    dataset.increase_max_skip()
            else:
                trainloader.dataset.increase_max_skip()

        # save model
        # 当 loss 小于 minloss 时保存检查点
        is_best = train_loss <= minloss
        minloss = min(minloss, train_loss)
        skips = [ds.max_skip for ds in trainloader.dataset.datasets] \
                if isinstance(trainloader.dataset, data.ConcatDataset) \
                 else trainloader.dataset.max_skip

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'loss': train_loss,
            'minloss': minloss,
            'optimizer': optimizer.state_dict(),
            'max_skip': skips,
        }, epoch + 1, is_best, checkpoint=opt.checkpoint, filename=opt.mode)

    logger.close()

    print('minimum loss:')
    print(minloss)

def train(trainloader, model, criterion, optimizer, epoch, use_cuda, iter_size, mode, threshold):
    # switch to train mode

    data_time = AverageMeter()
    loss = AverageMeter()

    end = time.time()

    bar = Bar('Processing', max=len(trainloader))
    optimizer.zero_grad()

    for batch_idx, data in enumerate(trainloader):
        frames, masks, objs, _ = data
        # measure data loading time
        data_time.update(time.time() - end)
        
        if use_cuda:
            frames = frames.cuda()
            masks = masks.cuda()

        N, T, C, H, W = frames.size()
        max_obj = masks.shape[2]-1

        total_loss = 0.0
        for idx in range(N):
            frame = frames[idx]
            mask = masks[idx]
            num_objects = objs[idx]

            keys = []
            vals = []
            for t in range(1, T):
                # memorize
                if t-1 == 0 or mode == 'mask':
                    tmp_mask = mask[t-1:t]
                elif mode == 'recurrent':
                    tmp_mask = out
                else:
                    pred_mask = out[0, 1:num_objects+1]
                    iou = mask_iou(pred_mask, mask[t-1, 1:num_objects+1])

                    if iou > threshold:
                        tmp_mask = out
                    else:
                        tmp_mask = mask[t-1:t]

                key, val, _ = model(frame=frame[t-1:t, :, :, :], mask=tmp_mask, num_objects=num_objects)
                keys.append(key)
                vals.append(val)

                # segment
                tmp_key = torch.cat(keys, dim=1)
                tmp_val = torch.cat(vals, dim=1)
                logits, ps = model(frame=frame[t:t+1, :, :, :], keys=tmp_key, values=tmp_val, num_objects=num_objects, max_obj=max_obj)

                out = torch.softmax(logits, dim=1)
                gt = mask[t:t+1]
                
                total_loss = total_loss + criterion(out, gt, num_objects)

        total_loss = total_loss / (N * (T-1))

        # record loss
        if total_loss.item() > 0.0:
            loss.update(total_loss.item(), 1)

        # compute gradient and do SGD step (divided by accumulated steps)
        total_loss /= iter_size
        total_loss.backward()

        if (batch_idx+1) % iter_size == 0:
            optimizer.step()
            model.zero_grad()

        # measure elapsed time
        end = time.time()
        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s |Loss: {loss:.5f}'.format(
            batch=batch_idx + 1,
            size=len(trainloader),
            data=data_time.val,
            loss=loss.val
        )
        bar.next()
    bar.finish()

    return loss.avg

def test(testloader, model, criterion, epoch, use_cuda, opt):

    data_time = AverageMeter()

    bar = Bar('Processing', max=len(testloader))

    with torch.no_grad():
        for batch_idx, data in enumerate(testloader):

            frames, masks, objs, infos = data

            if use_cuda:
                frames = frames.cuda()
                masks = masks.cuda()
                
            frames = frames[0]
            masks = masks[0]
            num_objects = objs[0]
            info = infos[0]
            max_obj = masks.shape[1]-1
            # compute output
            t1 = time.time()

            T, _, H, W = frames.shape
            pred = [masks[0:1]]
            keys = []
            vals = []
            for t in range(1, T):
                if t-1 == 0:
                    tmp_mask = masks[0:1]
                elif 'frame' in info and t-1 in info['frame']:
                    # start frame
                    mask_id = info['frame'].index(t-1)
                    tmp_mask = masks[mask_id:mask_id+1]
                    num_objects = max(num_objects, tmp_mask.max())
                else:
                    tmp_mask = out

                # memorize
                key, val, _ = model(frame=frames[t-1:t, :, :, :], mask=tmp_mask, num_objects=num_objects)

                # segment
                tmp_key = torch.cat(keys+[key], dim=1)
                tmp_val = torch.cat(vals+[val], dim=1)
                logits, ps = model(frame=frames[t:t+1, :, :, :], keys=tmp_key, values=tmp_val, num_objects=num_objects, max_obj=max_obj)

                out = torch.softmax(logits, dim=1)
                pred.append(out)

                if (t-1) % opt.save_freq == 0:
                    keys.append(key)
                    vals.append(val)
            
            pred = torch.cat(pred, dim=0)
            pred = pred.detach().cpu().numpy()
            write_mask(pred, info, opt, directory=opt.output_dir)

            toc = time.time() - t1

            data_time.update(toc, 1)
           
            # plot progress
            bar.suffix  = '({batch}/{size}) Time: {data:.3f}s'.format(
                batch=batch_idx + 1,
                size=len(testloader),
                data=data_time.sum
            )
            bar.next()
        bar.finish()

    return

if __name__ == '__main__':
    main()
