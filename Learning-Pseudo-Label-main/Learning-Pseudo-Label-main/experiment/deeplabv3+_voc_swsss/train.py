# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

import os
import sys
import time
import copy
import torch
import random
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms

from PIL import Image
from tqdm import tqdm
from config import config_dict
from lib.utils.imutils import img_denorm
from lib.utils.finalprocess import writelog
from lib.utils.visualization import generate_vis
from lib.utils.configuration import Configuration
from lib.net.generateNet import generate_net
from lib.net.loss import ClassLogSoftMax, SegLoss
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tensorboardX import SummaryWriter
from lib.datasets.generateData import generate_dataset

torch.manual_seed(1) # cpu
torch.cuda.manual_seed_all(1) #gpu
np.random.seed(1) #numpy
random.seed(1) #random and transforms
torch.backends.cudnn.deterministic=True # cudnn

def fuse_sample(sample_seg, sample_cls):#拼接
    sample = dict()
    for k in sample_seg:
        v1 = sample_seg[k]
        if type(v1) is torch.Tensor and k in sample_cls:
            v2 = sample_cls[k]
            v = torch.cat([v1,v2], dim=0)
            sample[k] = v
    return sample

def train_net(cfg, comment=''):
    period = 'train'
    transform = 'weak' if cfg.DATA_AUGTIME<=0 else 'strong'
    seg_dataset = generate_dataset(cfg, period=period, transform=transform)
    cfg2 = copy.deepcopy(cfg)
    #实现随机复制粘贴操作，一个样本的部分像素复制到另一个样本中
    cfg2.DATA_RANDOMCOPYPASTE = 0
    # 确定要图像增强操作的个数
    cfg2.DATA_AUGTIME = 0

    cls_dataset = generate_dataset(cfg2, period=period, transform=transform)
    # 用于初始化数据加载器的函数，设置随机种子
    def worker_init_fn(worker_id):
        np.random.seed(1 + worker_id)
    # 数据集分割的索引，强标记子集的大小
    split_idx = cfg.DATA_SPLIT
    seg_indices = [i%split_idx for i in range(len(cls_dataset))]
    cls_indices = list(range(len(cls_dataset)))
    # 创建数据集的采样器，用于随机采样数据
    seg_sampler = SubsetRandomSampler(seg_indices)
    cls_sampler = SubsetRandomSampler(cls_indices)
    # 子批次大小，分为强弱数据集
    subbatch = cfg.TRAIN_BATCHES//2
    seg_dataloader = DataLoader(seg_dataset, 
                batch_size=subbatch, 
                sampler=seg_sampler,
                num_workers=cfg.DATA_WORKERS,
                drop_last=True,
                worker_init_fn=worker_init_fn)
    cls_dataloader = DataLoader(cls_dataset, 
                batch_size=cfg.TRAIN_BATCHES-subbatch, 
                sampler=cls_sampler,
                num_workers=cfg.DATA_WORKERS,
                drop_last=True,
                worker_init_fn=worker_init_fn)
    
    HybridNet = generate_net(cfg, dilated=cfg.MODEL_BACKBONE_DILATED, multi_grid=cfg.MODEL_BACKBONE_MULTIGRID, deep_base=cfg.MODEL_BACKBONE_DEEPBASE)
    PseudoNet = generate_net(cfg, dilated=cfg.MODEL_BACKBONE_DILATED, multi_grid=cfg.MODEL_BACKBONE_MULTIGRID, deep_base=cfg.MODEL_BACKBONE_DEEPBASE)
    # 如果指定了训练的检查点，加载预训练模型的参数到混合网络上来
    if cfg.TRAIN_CKPT:
        HybridNet.load_state_dict(torch.load(cfg.TRAIN_CKPT),strict=False)
    # 写日志
    if cfg.TRAIN_TBLOG:
        from tensorboardX import SummaryWriter
        # Set the Tensorboard logger
        tblogger = SummaryWriter(cfg.LOG_DIR)    

    #print('Use %d GPU'%cfg.GPUS)
    device = torch.device(0)
    if cfg.GPUS > 1:
        # 实现多GPU并行计算
        HybridNet = nn.DataParallel(HybridNet)
        parameter_source_hybridnet = HybridNet.module
        PseudoNet = nn.DataParallel(PseudoNet)
        parameter_source_pseudonet = PseudoNet.module
    else:
        # 获取网络参数
        parameter_source_hybridnet = HybridNet
        parameter_source_pseudonet = PseudoNet
        
    HybridNet.to(device)        
    PseudoNet.to(device)        
    lr_rate = cfg.TRAIN_LR_RATE
    # 设置不同参数组的学习率、动量、权重衰减
    HybridNet_optimizer = optim.SGD(
        params = [
            {'params': get_params(parameter_source_hybridnet, key='backbone'), 'lr': lr_rate[0]*cfg.TRAIN_LR},
            {'params': get_params(parameter_source_hybridnet, key='others'),   'lr': lr_rate[1]*cfg.TRAIN_LR},
            {'params': get_params(parameter_source_hybridnet, key='cls'),      'lr': lr_rate[2]*cfg.TRAIN_LR},
        ],
        momentum=cfg.TRAIN_MOMENTUM,
        weight_decay=cfg.TRAIN_WEIGHT_DECAY
    )
    PseudoNet_optimizer = optim.SGD(
        params = [
            {'params': get_params(parameter_source_pseudonet, key='backbone'), 'lr': lr_rate[0]*cfg.TRAIN_LR},
            {'params': get_params(parameter_source_pseudonet, key='others'),   'lr': lr_rate[1]*cfg.TRAIN_LR},
            {'params': get_params(parameter_source_pseudonet, key='cls'),      'lr': lr_rate[2]*cfg.TRAIN_LR},
        ],
        momentum=cfg.TRAIN_MOMENTUM,
        weight_decay=cfg.TRAIN_WEIGHT_DECAY
    )
    # 计算迭代次数与epoch
    itr = cfg.TRAIN_MINEPOCH * len(cls_dataset)//(cfg.TRAIN_BATCHES-subbatch)
    max_itr = cfg.TRAIN_ITERATION
    max_epoch = max_itr*(cfg.TRAIN_BATCHES-subbatch)//len(cls_dataset)+1

    # 用于将训练日志信息写入到指定的日志目录
    tblogger = SummaryWriter(cfg.LOG_DIR)
    # 创建一个损失函数
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    # 用于混合精度训练，可以自动缩放梯度值，提高训练效率和减少内存占用
    scaler = torch.cuda.amp.GradScaler()
    score = -1
    score_mom = 0.99
    iou_t = torch.ones(cfg.MODEL_NUM_CLASSES).to(0)
    iou_p = torch.zeros(cfg.MODEL_NUM_CLASSES).to(0)
    iou_tp = torch.zeros(cfg.MODEL_NUM_CLASSES).to(0)
    # 显示当前迭代的进度，tqdm是创建进度条的库
    with tqdm(total=max_itr) as pbar:
        for epoch in range(cfg.TRAIN_MINEPOCH, max_epoch):
            # 遍历数据
            seg_enumerate = enumerate(seg_dataloader)
            cls_enumerate = enumerate(cls_dataloader)
            for i_batch in range(len(cls_dataset)//(cfg.TRAIN_BATCHES-subbatch)):
                _, sample_seg = next(seg_enumerate)
                _, sample_cls = next(cls_enumerate)
                # 融合得到一个新的样本
                sample = fuse_sample(sample_seg, sample_cls)
                # 调整学习率
                now_lr = adjust_lr(HybridNet_optimizer, itr, max_itr, cfg.TRAIN_LR, cfg.TRAIN_POWER, lr_rate)
                # 网络梯度置零
                HybridNet_optimizer.zero_grad()
                PseudoNet_optimizer.zero_grad()
                # 将图像、分割标签、掩码放到GPU上
                image = sample['image'].to(0)
                seg_label = sample['segmentation'].to(0)
                mask = sample['mask'].to(0)
                # 计算掩码的和并进行阈值化，得到二值掩码
                mask = torch.sum(mask.flatten(-2),dim=-1)
                mask[mask>0] = 1
                mask = mask.view(-1,1,1)
                onehot_label = sample['segmentation_onehot'].to(0)
                category = sample['category'].to(0)
                n,c,h,w = image.size()
                # 混合精度训练的上下文管理器
                with torch.cuda.amp.autocast():
                    pred_hybrid = HybridNet(image)
                    pred_pseudo = PseudoNet(image)
                    # 进行自适应最大池化操作，得到全局预测结果，大小调整为1*1
                    pred_global = F.adaptive_max_pool2d(pred_hybrid, (1,1))
                    # 主要为了类别预测，将pred_hybrid通过ClassLogSoftMax进行分类预测得到：
                    # 类别的softmax概率：clssoftmax_hybrid，类别对数softmax概率：clslogsoftmax_hybrid
                    clssoftmax_hybrid, clslogsoftmax_hybrid = ClassLogSoftMax(pred_hybrid,category)
                    # 根据类别的softmax概率在每个像素位置上选择概率最大的类别作为伪标签，使用detach将其从计算图中分离，以便后续操作不会影响梯度传播
                    pseudo_label = torch.argmax(clssoftmax_hybrid,dim=1).detach()
                    # 对部分样本应用掩码操作，得到最终的伪标签
                    pseudo_label[subbatch:] = pseudo_label[subbatch:] * (1-mask[subbatch:]) \
                                  + seg_label[subbatch:] * mask[subbatch:]

                    # 同上，对伪网络的操作
                    clssoftmax_pseudo, clslogsoftmax_pseudo = ClassLogSoftMax(pred_pseudo,category)
                    pseudo_label2 = torch.argmax(clssoftmax_pseudo,dim=1).detach()
                    pseudo_label2[subbatch:] = pseudo_label2[subbatch:] * (1-mask[subbatch:]) \
                                   + seg_label[subbatch:] * mask[subbatch:]
                    # 计算损失
                    loss_cls_hybrid = F.multilabel_soft_margin_loss(pred_global[:,1:,:,:],
                                            category[:,1:,:,:])
                    loss_seg_hybrid = SegLoss(clslogsoftmax_hybrid[:subbatch], seg_label[:subbatch])
                    loss_seg_pseudo = SegLoss(clslogsoftmax_pseudo[subbatch:], pseudo_label[subbatch:])

                    # Progressive Cross Training
                    #191-208   计算iou
                    cls_t,count_t = torch.unique(seg_label[:subbatch], sorted=True, return_counts=True)
                    if cls_t[-1] == 255:
                        cls_t = cls_t[:-1]
                        count_t = count_t[:-1]
                    cls_p,count_p = torch.unique(pseudo_label2[:subbatch], sorted=True, return_counts=True)
                    iou_label = seg_label[:subbatch].clone()
                    iou_label[seg_label[:subbatch] != pseudo_label2[:subbatch]] = 255
                    cls_tp, count_tp = torch.unique(iou_label, sorted=True, return_counts=True)
                    if cls_tp[-1] == 255:
                        cls_tp = cls_tp[:-1]
                        count_tp = count_tp[:-1]
                    iou_t *= score_mom
                    iou_p *= score_mom
                    iou_tp *= score_mom
                    iou_t[cls_t] += count_t
                    iou_p[cls_p] += count_p
                    iou_tp[cls_tp] += count_tp
                    iou = iou_tp/(iou_t+iou_p-iou_tp)

                    loss_pct = SegLoss(clslogsoftmax_hybrid[subbatch:], pseudo_label2[subbatch:], reduction='none')
                    sample_weight = category[subbatch:] * iou.repeat(n-subbatch,1).unsqueeze(-1).unsqueeze(-1)
                    sample_weight[sample_weight==0] = 1
                    sample_weight = torch.min(sample_weight,dim=1,keepdim=False)[0]
                    sample_weight = torch.pow(sample_weight,cfg.TRAIN_SCORE_POW)
                    loss_pct = torch.mean(loss_pct*sample_weight)
                    # 四个损失
                    loss = loss_seg_hybrid + loss_seg_pseudo + loss_cls_hybrid + loss_pct
                # 反向传播和优化器更新
                scaler.scale(loss).backward()
                scaler.step(HybridNet_optimizer)
                scaler.step(PseudoNet_optimizer)
                scaler.update()
                #更新进度条的描述信息并更新进度条的进度
                pbar.set_description("%s, loss-%.3f" % (comment,loss.item()))
                pbar.update(1)
                time.sleep(0.001)
                #  用于生成可视化结果并将其添加到TensorBoard日志中
                if cfg.TRAIN_TBLOG and itr%100 == 0:
                    images = img_denorm(image[0].cpu().numpy()).astype(np.uint8)
                    imagew = img_denorm(image[-1].cpu().numpy()).astype(np.uint8)
                    imgcolor = np.stack((images,imagew),axis=0)

                    seg_labels = seg_label[0].cpu().numpy()
                    labels_colors = cls_dataset.label2colormap(seg_labels).transpose((2,0,1))
                    seg_labelw = seg_label[-1].cpu().numpy()
                    labels_colorw = cls_dataset.label2colormap(seg_labelw).transpose((2,0,1))
                    label_color = np.stack((labels_colors,labels_colorw),axis=0)

                    hybrid_s = vis_pred(clssoftmax_hybrid[0], cls_dataset.label2colormap)
                    hybrid_w = vis_pred(clssoftmax_hybrid[-1], cls_dataset.label2colormap)
                    pseudo_s = vis_pred(clssoftmax_pseudo[0], cls_dataset.label2colormap)
                    pseudo_w = vis_pred(clssoftmax_pseudo[-1], cls_dataset.label2colormap)
                    pred_color = np.stack((hybrid_s, hybrid_w, pseudo_s, pseudo_w),axis=0)

                    tblogger.add_scalar('loss', loss.item(), itr)
                    tblogger.add_scalar('lr', now_lr, itr)
                    tblogger.add_scalar('loss_pct', loss_pct.item(), itr)
                    tblogger.add_images('Input', imgcolor, itr)
                    tblogger.add_images('Label', label_color, itr)
                    tblogger.add_images('Seg', pred_color, itr)
                itr += 1
                if itr>=max_itr:
                    break
            filename_prototype = f'{cfg.MODEL_NAME}_{cfg.MODEL_BACKBONE}_' \
                                 f'{cfg.DATA_NAME}_epoch{epoch}_{comment}'
            save_path = os.path.join(cfg.MODEL_SAVE_DIR,f'{filename_prototype}_hybridnet.pth')
            torch.save(parameter_source_hybridnet.state_dict(), save_path)

            save_path = os.path.join(cfg.MODEL_SAVE_DIR,f'{filename_prototype}_pseudonet.pth')
            torch.save(parameter_source_pseudonet.state_dict(), save_path)
    filename_prototype = f'{cfg.MODEL_NAME}_{cfg.MODEL_BACKBONE}_' \
                         f'{cfg.DATA_NAME}_itr{cfg.TRAIN_ITERATION}_' \
                         f'sub{cfg.DATA_SPLIT}_{comment}'
    save_path = os.path.join(cfg.MODEL_SAVE_DIR,f'{filename_prototype}_hybridnet.pth')
    torch.save(parameter_source_hybridnet.state_dict(),save_path)
    save_path = os.path.join(cfg.MODEL_SAVE_DIR,f'{filename_prototype}_pseudonet.pth')
    torch.save(parameter_source_pseudonet.state_dict(),save_path)
    print('%s has been saved'%save_path)
    if cfg.TRAIN_TBLOG:
        tblogger.close()
    writelog(cfg, period)

def adjust_lr(optimizer, itr, max_itr, lr_init, power, lr_rate):
    now_lr = lr_init * ((1 - itr/(max_itr+1e-5)) ** power) 
    optimizer.param_groups[0]['lr'] = now_lr * lr_rate[0]
    optimizer.param_groups[1]['lr'] = now_lr * lr_rate[1]
    optimizer.param_groups[2]['lr'] = now_lr * lr_rate[2]
    return now_lr

def get_params(model, key):
    for m in model.named_modules():
        if key == 'backbone':
            #if ('backbone' in m[0]) and isinstance(m[1], (nn.Conv2d, SynchronizedBatchNorm2d)):
            if ('backbone' in m[0]) and isinstance(m[1], (nn.Conv2d, nn.BatchNorm2d)):
                for p in m[1].parameters():
                    yield p
        elif key == 'cls':
            if ('cls_conv' in m[0]) and isinstance(m[1], (nn.Conv2d, nn.BatchNorm2d)):
                for p in m[1].parameters():
                    yield p
        elif key == 'others':
            if ('backbone' not in m[0] and 'cls_conv' not in m[0]) and isinstance(m[1], (nn.Conv2d, nn.BatchNorm2d)):
                for p in m[1].parameters():
                    yield p

def vis_pred(pred, colorfunc):
    pred = torch.argmax(pred, dim=0)
    pred_color = colorfunc(pred.cpu().numpy()).transpose((2,0,1))
    return pred_color

if __name__ == '__main__':
    cfg = Configuration(config_dict)
    train_net(cfg)
