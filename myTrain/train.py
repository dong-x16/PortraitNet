import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.utils.data
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.nn import DataParallel

import numpy as np
import argparse
import time
import os
import shutil
from easydict import EasyDict as edict
from yaml import load

import sys
sys.path.append('../data/')
sys.path.append('../util/')
sys.path.append('../model/')

import datasets
from datasets import Human
from data_aug import Normalize_Img, Anti_Normalize_Img
from focal_loss import FocalLoss

from logger import Logger
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def calcIOU(img, mask):
    sum1 = img + mask
    sum1[sum1>0] = 1
    sum2 = img + mask
    sum2[sum2<2] = 0
    sum2[sum2>=2] = 1
    if np.sum(sum1) == 0:
        return 1
    else:
        return 1.0*np.sum(sum2)/np.sum(sum1)

def get_parameters(model, args, useDeconvGroup=True):
    lr_0 = []
    lr_1 = []
    params_dict = dict(model.named_parameters())
    for key, value in params_dict.items():
        if 'deconv' in key and useDeconvGroup==True:
            print ("useDeconvGroup=True, lr=0, key: ", key)
            lr_0.append(value)
        else:
            lr_1.append(value)
    params = [{'params': lr_0, 'lr': args.lr * 0},
              {'params': lr_1, 'lr': args.lr * 1}]
    return params, [0., 1.]

def adjust_learning_rate(optimizer, epoch, args, multiple):
    """Sets the learning rate to the initial LR decayed by 0.95 every 20 epochs"""
    # lr = args.lr * (0.95 ** (epoch // 4))
    lr = args.lr * (0.95 ** (epoch // 20))
    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lr * multiple[i]
    pass

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    pass

def loss_KL(student_outputs, teacher_outputs, T):
    """
    Code referenced from: 
    https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py
    
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    KD_loss = nn.KLDivLoss()(F.log_softmax(student_outputs/T, dim=1), 
                             F.softmax(teacher_outputs/T, dim=1)) * T * T
    return KD_loss


def test(dataLoader, netmodel, optimizer, epoch, logger, exp_args):
    batch_time = AverageMeter('batch_time')
    data_time = AverageMeter('data_time')
    losses = AverageMeter('losses')
    
    losses_mask_ori = AverageMeter('losses_mask_ori')
    losses_mask = AverageMeter('losses_mask')
    
    losses_edge_ori = AverageMeter('losses_edge_ori')
    losses_edge = AverageMeter('losses_edge')
    
    losses_stability_mask = AverageMeter('losses_stability_mask')
    losses_stability_edge = AverageMeter('losses_stability_edge')
    
    # switch to eval mode
    netmodel.eval()
    
    loss_Softmax = nn.CrossEntropyLoss(ignore_index=255) # mask loss
    loss_Focalloss = FocalLoss(gamma=2) # edge loss
    # loss_l2 = nn.MSELoss() # edge loss
    
    end = time.time()
    softmax = nn.Softmax(dim=1)
    iou = 0
    
    for i, (input_ori, input, edge, mask) in enumerate(dataLoader):  
        data_time.update(time.time() - end)
        input_ori_var = Variable(input_ori.cuda())
        input_var = Variable(input.cuda())
        edge_var = Variable(edge.cuda())
        mask_var = Variable(mask.cuda())
        
        if exp_args.addEdge == True:
            output_mask, output_edge = netmodel(input_var)
            loss_mask = loss_Softmax(output_mask, mask_var)
            losses_mask.update(loss_mask.data[0], input.size(0))
            # loss_edge = loss_l2(output_edge, edge_var) * exp_args.edgeRatio
            loss_edge = loss_Focalloss(output_edge, edge_var) * exp_args.edgeRatio
            losses_edge.update(loss_edge.data[0], input.size(0))
            loss = loss_mask + loss_edge
            
            if exp_args.stability == True:
                output_mask_ori, output_edge_ori = netmodel(input_ori_var)
                loss_mask_ori = loss_Softmax(output_mask_ori, mask_var)
                losses_mask_ori.update(loss_mask_ori.data[0], input.size(0))
                # loss_edge_ori = loss_l2(output_edge_ori, edge_var) * exp_args.edgeRatio
                loss_edge_ori = loss_Focalloss(output_edge_ori, edge_var) * exp_args.edgeRatio
                losses_edge_ori.update(loss_edge_ori.data[0], input.size(0))
                
                if exp_args.use_kl == False:
                    # consistency constraint loss: L2 distance 
                    loss_stability_mask = loss_l2(output_mask, 
                                                  Variable(output_mask_ori.data, 
                                                           requires_grad = False)) * exp_args.alpha
                    loss_stability_edge = loss_l2(output_edge, 
                                                  Variable(output_edge_ori.data, 
                                                           requires_grad = False)) * exp_args.alpha * exp_args.edgeRatio
                else:
                    # consistency constraint loss: KL distance
                    loss_stability_mask = loss_KL(output_mask, 
                                                  Variable(output_mask_ori.data, requires_grad = False), 
                                                  exp_args.temperature) * exp_args.alpha
                    loss_stability_edge = loss_KL(output_edge, 
                                                  Variable(output_edge_ori.data, requires_grad = False), 
                                                  exp_args.temperature) * exp_args.alpha * exp_args.edgeRatio
                    
                losses_stability_mask.update(loss_stability_mask.data[0], input.size(0))
                losses_stability_edge.update(loss_stability_edge.data[0], input.size(0))
                
                # total loss
                # loss = loss_mask + loss_mask_ori + loss_edge + loss_edge_ori + loss_stability_mask + loss_stability_edge
                loss = loss_mask + loss_mask_ori + loss_stability_mask + loss_edge
        else:
            output_mask = netmodel(input_var)
            loss_mask = loss_Softmax(output_mask, mask_var)
            losses_mask.update(loss_mask.data[0], input.size(0))
            loss = loss_mask
            
            if exp_args.stability == True:
                # loss part2
                output_mask_ori = netmodel(input_ori_var)
                loss_mask_ori = loss_Softmax(output_mask_ori, mask_var)
                losses_mask_ori.update(loss_mask_ori.data[0], input.size(0))
                if exp_args.use_kl == False:
                    # consistency constraint loss: L2 distance
                    loss_stability_mask = loss_l2(output_mask, 
                                                  Variable(output_mask_ori.data, 
                                                           requires_grad = False)) * exp_args.alpha
                else:
                    # consistency constraint loss: KL distance
                    loss_stability_mask = loss_KL(output_mask, 
                                                  Variable(output_mask_ori.data, requires_grad = False),
                                                  exp_args.temperature) * exp_args.alpha
                losses_stability_mask.update(loss_stability_mask.data[0], input.size(0))
                # total loss
                loss = loss_mask + loss_mask_ori + loss_stability_mask
        
        # total loss
        loss = loss_mask
        losses.update(loss.data[0], input.size(0))
        
        prob = softmax(output_mask)[0,1,:,:]
        pred = prob.data.cpu().numpy()
        pred[pred>0.5] = 1
        pred[pred<=0.5] = 0
        iou += calcIOU(pred, mask_var[0].data.cpu().numpy())
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.printfreq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Lr-deconv: [{3}]\t'
                  'Lr-other: [{4}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                      epoch, i, len(dataLoader), 
                      optimizer.param_groups[0]['lr'],
                      optimizer.param_groups[1]['lr'], 
                      loss=losses)) 
            
        ## '===========> logger <==========='
        # (1) Log the scalar values
        if exp_args.addEdge == True and exp_args.stability == True:
            info = { # batch_time.name: batch_time.val,
                     # data_time.name: data_time.val,
                     losses.name: losses.val,
                     losses_mask_ori.name: losses_mask_ori.val,
                     losses_mask.name: losses_mask.val,
                     losses_edge_ori.name: losses_edge_ori.val,
                     losses_edge.name: losses_edge.val,
                     losses_stability_mask.name: losses_stability_mask.val, 
                     losses_stability_edge.name: losses_stability_edge.val
                   }
        elif exp_args.addEdge == True and exp_args.stability == False:
            info = { # batch_time.name: batch_time.val,
                     # data_time.name: data_time.val,
                     losses.name: losses.val,
                     losses_mask.name: losses_mask.val,
                     losses_edge.name: losses_edge.val,
                   }
        elif exp_args.addEdge == False and exp_args.stability == True:
            info = { # batch_time.name: batch_time.val,
                     # data_time.name: data_time.val,
                     losses.name: losses.val,
                     losses_mask_ori.name: losses_mask_ori.val,
                     losses_mask.name: losses_mask.val,
                     losses_stability_mask.name: losses_stability_mask.val, 
                   }
        elif exp_args.addEdge == False and exp_args.stability == False:
            info = { # batch_time.name: batch_time.val,
                     # data_time.name: data_time.val,
                     losses.name: losses.val,
                     losses_mask.name: losses_mask.val,
                   }
            
        for tag, value in info.items():
            logger.scalar_summary(tag, value, step=i)
        '''
        # (2) Log values and gradients of the parameters (histogram)
        for tag, value in netmodel.named_parameters():
            tag = tag.replace('.', '/')
            logger.histo_summary(tag, value.data.cpu().numpy(), step=i)
            if value.grad is None:
                continue
            logger.histo_summary(tag+'/grad', value.grad.cpu().data.numpy(), step=i)
            break
        '''
        # (3) Log the images
        if i % (args.printfreq) == 0:
            num = 2
            input_img = np.uint8((
                        Anti_Normalize_Img(
                            np.transpose(input.cpu().numpy()[0:num], (0, 2, 3, 1)), 
                            scale=exp_args.img_scale, 
                            mean=exp_args.img_mean, 
                            val=exp_args.img_val)))[:,:,:,:3][:,:,:,::-1]
            
            if exp_args.video == True:
                input_prior = np.float32(np.transpose(input.cpu().numpy()[0:num], (0, 2, 3, 1))[:,:,:,3])
            
            input_mask = mask.cpu().numpy()[0:num]
            input_mask[input_mask==255] = 0
            softmax = nn.Softmax(dim=1)
            prob = softmax(output_mask)
            masks_pred = np.transpose(prob.data.cpu().numpy()[0:num], (0, 2, 3, 1))[:,:,:,1]
            
            info = {}
            info['input_img'] = input_img
            if exp_args.video == True:
                info['input_prior'] = input_prior*255
            info['input_mask'] = input_mask*255
            info['output_mask'] = masks_pred*255

            if exp_args.addEdge == True:
                input_edge = edge.cpu().numpy()[0:num]
                edge_pred = np.transpose(output_edge.data.cpu().numpy()[0:num], (0, 2, 3, 1))[:,:,:,0]
                
                if exp_args.stability == True:
                    input_img_ori = np.uint8((
                        Anti_Normalize_Img(
                            np.transpose(input_ori.cpu().numpy()[0:num], (0, 2, 3, 1)), 
                            scale=exp_args.img_scale, 
                            mean=exp_args.img_mean, 
                            val=exp_args.img_val)))[:,:,:,:3][:,:,:,::-1]
             
                    prob_ori = softmax(output_mask_ori)
                    masks_pred_ori = np.transpose(prob_ori.data.cpu().numpy()[0:num], (0, 2, 3, 1))[:,:,:,1]
                    edge_pred_ori = np.transpose(output_edge_ori.data.cpu().numpy()[0:num], (0, 2, 3, 1))[:,:,:,0]
                    
                    info['input_img_ori'] = input_img_ori
                    info['output_mask_ori'] = masks_pred_ori*255
                    
                    info['input_edge'] = input_edge*255
                    info['output_edge'] = edge_pred*255
                    info['output_edge_ori'] = edge_pred_ori*255
                else:
                    info['input_edge'] = input_edge*255
                    info['output_edge'] = edge_pred*255
            else:
                if exp_args.stability == True:
                    input_img_ori = np.uint8((
                        Anti_Normalize_Img(
                            np.transpose(input_ori.cpu().numpy()[0:num], (0, 2, 3, 1)), 
                            scale=exp_args.img_scale, 
                            mean=exp_args.img_mean, 
                            val=exp_args.img_val)))[:,:,:,:3][:,:,:,::-1]
             
                    prob_ori = softmax(output_mask_ori)
                    masks_pred_ori = np.transpose(prob_ori.data.cpu().numpy()[0:num], (0, 2, 3, 1))[:,:,:,1]
                    
                    info['input_img_ori'] = input_img_ori
                    info['output_mask_ori'] = masks_pred_ori*255
            print (np.max(masks_pred), np.min(masks_pred))
            

            for tag, images in info.items():
                logger.image_summary(tag, images, step=i)
            
    # return losses.avg
    return 1-iou/len(dataLoader)

def train(dataLoader, netmodel, optimizer, epoch, logger, exp_args):
    batch_time = AverageMeter('batch_time')
    data_time = AverageMeter('data_time')
    
    losses = AverageMeter('losses')
    losses_mask = AverageMeter('losses_mask')
    
    if exp_args.addEdge == True:
        losses_edge_ori = AverageMeter('losses_edge_ori')
        losses_edge = AverageMeter('losses_edge')
    
    if exp_args.stability == True:
        losses_mask_ori = AverageMeter('losses_mask_ori')
        losses_stability_mask = AverageMeter('losses_stability_mask')
        losses_stability_edge = AverageMeter('losses_stability_edge')

    netmodel.train() # switch to train mode
    
    loss_Softmax = nn.CrossEntropyLoss(ignore_index=255) # mask loss
    # in our experiments, focalloss is better than l2 loss
    loss_Focalloss = FocalLoss(gamma=2) # boundary loss
    # loss_l2 = nn.MSELoss() # boundary loss
    
    end = time.time()
    for i, (input_ori, input, edge, mask) in enumerate(dataLoader):  
        data_time.update(time.time() - end)
        input_ori_var = Variable(input_ori.cuda())
        input_var = Variable(input.cuda())
        edge_var = Variable(edge.cuda())
        mask_var = Variable(mask.cuda())
        
        if exp_args.addEdge == True:
            output_mask, output_edge = netmodel(input_var)
            loss_mask = loss_Softmax(output_mask, mask_var)
            losses_mask.update(loss_mask.data[0], input.size(0))
            
            # loss_edge = loss_l2(output_edge, edge_var) * exp_args.edgeRatio
            loss_edge = loss_Focalloss(output_edge, edge_var) * exp_args.edgeRatio
            losses_edge.update(loss_edge.data[0], input.size(0))
            
            # total loss
            loss = loss_mask + loss_edge
            
            if exp_args.stability == True:
                output_mask_ori, output_edge_ori = netmodel(input_ori_var)
                loss_mask_ori = loss_Softmax(output_mask_ori, mask_var)
                losses_mask_ori.update(loss_mask_ori.data[0], input.size(0))
                
                # loss_edge_ori = loss_l2(output_edge_ori, edge_var) * exp_args.edgeRatio
                loss_edge_ori = loss_Focalloss(output_edge_ori, edge_var) * exp_args.edgeRatio
                losses_edge_ori.update(loss_edge_ori.data[0], input.size(0))
                
                # in our experiments, kl loss is better than l2 loss
                if exp_args.use_kl == False:
                    # consistency constraint loss: L2 distance 
                    loss_stability_mask = loss_l2(output_mask, 
                                                  Variable(output_mask_ori.data, 
                                                           requires_grad = False)) * exp_args.alpha
                    loss_stability_edge = loss_l2(output_edge, 
                                                  Variable(output_edge_ori.data, 
                                                           requires_grad = False)) * exp_args.alpha * exp_args.edgeRatio
                else:
                    # consistency constraint loss: KL distance (better than L2 distance)
                    loss_stability_mask = loss_KL(output_mask, 
                                                  Variable(output_mask_ori.data, requires_grad = False), 
                                                  exp_args.temperature) * exp_args.alpha
                    loss_stability_edge = loss_KL(output_edge, 
                                                  Variable(output_edge_ori.data, requires_grad = False), 
                                                  exp_args.temperature) * exp_args.alpha * exp_args.edgeRatio
                
                losses_stability_mask.update(loss_stability_mask.data[0], input.size(0))
                losses_stability_edge.update(loss_stability_edge.data[0], input.size(0))
                
                # total loss
                # loss = loss_mask + loss_mask_ori + loss_edge + loss_edge_ori + loss_stability_mask + loss_stability_edge
                loss = loss_mask + loss_mask_ori + loss_stability_mask + loss_edge
        else:
            output_mask = netmodel(input_var)
            loss_mask = loss_Softmax(output_mask, mask_var)
            losses_mask.update(loss_mask.data[0], input.size(0))
            # total loss: only include mask loss
            loss = loss_mask
            
            if exp_args.stability == True:
                output_mask_ori = netmodel(input_ori_var)
                loss_mask_ori = loss_Softmax(output_mask_ori, mask_var)
                losses_mask_ori.update(loss_mask_ori.data[0], input.size(0))
                if exp_args.use_kl == False:
                    # consistency constraint loss: L2 distance 
                    loss_stability_mask = loss_l2(output_mask, 
                                                  Variable(output_mask_ori.data, 
                                                           requires_grad = False)) * exp_args.alpha
                else:
                    # consistency constraint loss: KL distance (better than L2 distance)
                    loss_stability_mask = loss_KL(output_mask, 
                                                  Variable(output_mask_ori.data, requires_grad = False), 
                                                  exp_args.temperature) * exp_args.alpha
                losses_stability_mask.update(loss_stability_mask.data[0], input.size(0))
                
                # total loss
                loss = loss_mask + loss_mask_ori + loss_stability_mask
                
        losses.update(loss.data[0], input.size(0))
        
        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.printfreq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Lr-deconv: [{3}]\t'
                  'Lr-other: [{4}]\t'
                  # 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  # 'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                      epoch, i, len(dataLoader), 
                      optimizer.param_groups[0]['lr'], 
                      optimizer.param_groups[1]['lr'], 
                      loss=losses)) 
            
        ## '===========> logger <==========='
        # (1) Log the scalar values
        if exp_args.addEdge == True and exp_args.stability == True:
            info = { # batch_time.name: batch_time.val,
                     # data_time.name: data_time.val,
                     losses.name: losses.val,
                     losses_mask_ori.name: losses_mask_ori.val,
                     losses_mask.name: losses_mask.val,
                     losses_edge_ori.name: losses_edge_ori.val,
                     losses_edge.name: losses_edge.val,
                     losses_stability_mask.name: losses_stability_mask.val, 
                     losses_stability_edge.name: losses_stability_edge.val
                   }
        elif exp_args.addEdge == True and exp_args.stability == False:
            info = { # batch_time.name: batch_time.val,
                     # data_time.name: data_time.val,
                     losses.name: losses.val,
                     losses_mask.name: losses_mask.val,
                     losses_edge.name: losses_edge.val,
                   }
        elif exp_args.addEdge == False and exp_args.stability == True:
            info = { # batch_time.name: batch_time.val,
                     # data_time.name: data_time.val,
                     losses.name: losses.val,
                     losses_mask_ori.name: losses_mask_ori.val,
                     losses_mask.name: losses_mask.val,
                     losses_stability_mask.name: losses_stability_mask.val, 
                   }
        elif exp_args.addEdge == False and exp_args.stability == False:
            info = { # batch_time.name: batch_time.val,
                     # data_time.name: data_time.val,
                     losses.name: losses.val,
                     losses_mask.name: losses_mask.val,
                   }
        for tag, value in info.items():
            logger.scalar_summary(tag, value, step=i)
            
        '''
        # (2) Log values and gradients of the parameters (histogram)
        for tag, value in netmodel.named_parameters():
            tag = tag.replace('.', '/')
            logger.histo_summary(tag, value.data.cpu().numpy(), step=i)
            if value.grad is None:
                continue
            logger.histo_summary(tag+'/grad', value.grad.cpu().data.numpy(), step=i)
            break
        '''
        
        # (3) Log the images
        if i % (args.printfreq) == 0:
            num = 2
            input_img = np.uint8((
                        Anti_Normalize_Img(
                            np.transpose(input.cpu().numpy()[0:num], (0, 2, 3, 1)), 
                            scale=exp_args.img_scale, 
                            mean=exp_args.img_mean, 
                            val=exp_args.img_val)))[:,:,:,:3][:,:,:,::-1]
            
            if exp_args.video == True:
                input_prior = np.float32(np.transpose(input.cpu().numpy()[0:num], (0, 2, 3, 1))[:,:,:,3])
            
            input_mask = mask.cpu().numpy()[0:num]
            input_mask[input_mask==255] = 0
            softmax = nn.Softmax(dim=1)
            prob = softmax(output_mask)
            masks_pred = np.transpose(prob.data.cpu().numpy()[0:num], (0, 2, 3, 1))[:,:,:,1]
            
            info = {}
            info['input_img'] = input_img
            if exp_args.video == True:
                info['input_prior'] = input_prior*255
            info['input_mask'] = input_mask*255
            info['output_mask'] = masks_pred*255

            if exp_args.addEdge == True:
                input_edge = edge.cpu().numpy()[0:num]
                edge_pred = np.transpose(output_edge.data.cpu().numpy()[0:num], (0, 2, 3, 1))[:,:,:,0]
                
                if exp_args.stability == True:
                    input_img_ori = np.uint8((
                        Anti_Normalize_Img(
                            np.transpose(input_ori.cpu().numpy()[0:num], (0, 2, 3, 1)), 
                            scale=exp_args.img_scale, 
                            mean=exp_args.img_mean, 
                            val=exp_args.img_val)))[:,:,:,:3][:,:,:,::-1]
                    
                    prob_ori = softmax(output_mask_ori)
                    masks_pred_ori = np.transpose(prob_ori.data.cpu().numpy()[0:num], (0, 2, 3, 1))[:,:,:,1]
                    edge_pred_ori = np.transpose(output_edge_ori.data.cpu().numpy()[0:num], (0, 2, 3, 1))[:,:,:,0]
                    
                    info['input_img_ori'] = input_img_ori
                    info['output_mask_ori'] = masks_pred_ori*255
                    
                    info['input_edge'] = input_edge*255
                    info['output_edge'] = edge_pred*255
                    info['output_edge_ori'] = edge_pred_ori*255
                else:
                    info['input_edge'] = input_edge*255
                    info['output_edge'] = edge_pred*255
            else:
                if exp_args.stability == True:
                    input_img_ori = np.uint8((
                        Anti_Normalize_Img(
                            np.transpose(input_ori.cpu().numpy()[0:num], (0, 2, 3, 1)), 
                            scale=exp_args.img_scale, 
                            mean=exp_args.img_mean, 
                            val=exp_args.img_val)))[:,:,:,:3][:,:,:,::-1]
                    
                    prob_ori = softmax(output_mask_ori)
                    masks_pred_ori = np.transpose(prob_ori.data.cpu().numpy()[0:num], (0, 2, 3, 1))[:,:,:,1]
                    
                    info['input_img_ori'] = input_img_ori
                    info['output_mask_ori'] = masks_pred_ori*255
                
            print (np.max(masks_pred), np.min(masks_pred))
            
            for tag, images in info.items():
                logger.image_summary(tag, images, step=i)
        
    pass

def save_checkpoint(state, is_best, root, filename='checkpoint.pth.tar'):
    torch.save(state, root+filename)
    if is_best:
        shutil.copyfile(root+filename, root+'model_best.pth.tar')
        

def main(args):
    cudnn.benchmark = True
    assert args.model in ['PortraitNet', 'ENet', 'BiSeNet'], 'Error!, <model> should in [PortraitNet, ENet, BiSeNet]'
    
    config_path = args.config_path
    print ('===========> loading config <============')
    print ("config path: ", config_path)
    with open(config_path,'rb') as f:
        cont = f.read()
    cf = load(cont)
    
    print ('===========> loading data <===========')
    exp_args = edict()
    
    exp_args.istrain = cf['istrain'] # set the mode 
    exp_args.task = cf['task'] # only support 'seg' now
    exp_args.datasetlist = cf['datasetlist']
    exp_args.model_root = cf['model_root'] 
    exp_args.data_root = cf['data_root']
    exp_args.file_root = cf['file_root']

    # set log path
    logs_path = os.path.join(exp_args.model_root, 'log/')
    if os.path.exists(logs_path):
        shutil.rmtree(logs_path)
    logger_train = Logger(logs_path + 'train')
    logger_test = Logger(logs_path + 'test')
    
    # the height of input images, default=224
    exp_args.input_height = cf['input_height']
    # the width of input images, default=224
    exp_args.input_width = cf['input_width']
    
    # if exp_args.video=True, add prior channel for input images, default=False
    exp_args.video = cf['video']
    # the probability to set empty prior channel, default=0.5
    exp_args.prior_prob = cf['prior_prob']
    
    # whether to add boundary auxiliary loss, default=False
    exp_args.addEdge = cf['addEdge']
    # the weight of boundary auxiliary loss, default=0.1
    exp_args.edgeRatio = cf['edgeRatio']
    # whether to add consistency constraint loss, default=False
    exp_args.stability = cf['stability']
    # whether to use KL loss in consistency constraint loss, default=True
    exp_args.use_kl = cf['use_kl']
    # temperature in consistency constraint loss, default=1
    exp_args.temperature = cf['temperature'] 
    # the weight of consistency constraint loss, default=2
    exp_args.alpha = cf['alpha'] 
    
    # input normalization parameters
    exp_args.padding_color = cf['padding_color']
    exp_args.img_scale = cf['img_scale']
    # BGR order, image mean, default=[103.94, 116.78, 123.68]
    exp_args.img_mean = cf['img_mean']
    # BGR order, image val, default=[1/0.017, 1/0.017, 1/0.017]
    exp_args.img_val = cf['img_val'] 
    
    # whether to use pretian model to init portraitnet
    exp_args.init = cf['init'] 
    # whether to continue training
    exp_args.resume = cf['resume'] 
    
    # if exp_args.useUpsample==True, use nn.Upsample in decoder, else use nn.ConvTranspose2d
    exp_args.useUpsample = cf['useUpsample'] 
    # if exp_args.useDeconvGroup==True, set groups=input_channel in nn.ConvTranspose2d
    exp_args.useDeconvGroup = cf['useDeconvGroup'] 
    
    # set training dataset
    exp_args.istrain = True
    dataset_train = Human(exp_args)
    print ("image number in training: ", len(dataset_train))
    dataLoader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batchsize, 
                                                   shuffle=True, num_workers= args.workers)
    
    # set testing dataset
    exp_args.istrain = False
    dataset_test = Human(exp_args)
    print ("image number in testing: ", len(dataset_test))
    dataLoader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, 
                                                  shuffle=False, num_workers=args.workers)
    
    exp_args.istrain = True
    print ("finish load dataset ...")
    
    print ('===========> loading model <===========')
    
    if args.model == 'PortraitNet':
        # train our model: portraitnet
        import model_mobilenetv2_seg_small as modellib
        netmodel = modellib.MobileNetV2(n_class=2, 
                                        useUpsample=exp_args.useUpsample, 
                                        useDeconvGroup=exp_args.useDeconvGroup, 
                                        addEdge=exp_args.addEdge, 
                                        channelRatio=1.0, 
                                        minChannel=16, 
                                        weightInit=True,
                                        video=exp_args.video).cuda()
        print ("finish load PortraitNet ...")
        
    elif args.model == 'BiSeNet':
        # train BiSeNet
        import model_BiSeNet as modellib
        netmodel = modellib.BiSeNet(n_class=2, 
                                    useUpsample=exp_args.useUpsample, 
                                    useDeconvGroup=exp_args.useDeconvGroup, 
                                    addEdge=exp_args.addEdge).cuda()
        print ("finish load BiSeNet ...")
        
    elif args.model == 'ENet':
        # trian ENet
        import model_enet as modellib
        netmodel = modellib.ENet(n_class=2).cuda()
        print ("finish load ENet ...")
        
    # optimizer = torch.optim.SGD(netmodel.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weightdecay)  
    params, multiple = get_parameters(netmodel, args, useDeconvGroup=exp_args.useDeconvGroup)
    # optimizer = torch.optim.SGD(params, args.lr, momentum=args.momentum, weight_decay=args.weightdecay) 
    optimizer = torch.optim.Adam(params, args.lr, weight_decay=args.weightdecay) 
    
    if exp_args.init == True:
        pretrained_state_dict = torch.load('pretrained_mobilenetv2_base.pth')
        pretrained_state_dict_keys = pretrained_state_dict.keys()
        netmodel_state_dict = netmodel.state_dict()
        netmodel_state_dict_keys = netmodel.state_dict().keys()
        print ("pretrain keys: ", len(pretrained_state_dict_keys))
        print ("netmodel keys: ", len(netmodel_state_dict_keys))
        weights_load = {}
        for k in range(len(pretrained_state_dict_keys)):
            if pretrained_state_dict[pretrained_state_dict_keys[k]].shape == \
            netmodel_state_dict[netmodel_state_dict_keys[k]].shape:
                weights_load[netmodel_state_dict_keys[k]] = pretrained_state_dict[pretrained_state_dict_keys[k]]
                print ('init model', netmodel_state_dict_keys[k], 
                       'from pretrained', pretrained_state_dict_keys[k])
            else:
                break
        print ("init len is:", len(weights_load)) 
        netmodel_state_dict.update(weights_load)
        netmodel.load_state_dict(netmodel_state_dict)
        print ("load model init finish...")
    
    if exp_args.resume:
        bestModelFile = os.path.join(exp_args.model_root, 'model_best.pth.tar')
        if os.path.isfile(bestModelFile):
            checkpoint = torch.load(bestModelFile)
            netmodel.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            gap = checkpoint['epoch']
            minLoss = checkpoint['minLoss']
            print("=> loaded checkpoint '{}' (epoch {})".format(bestModelFile, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(bestModelFile))
    else:
        minLoss = 10000
        gap = 0
        
    for epoch in range(gap, 2000):
        adjust_learning_rate(optimizer, epoch, args, multiple)
        print ('===========>   training    <===========')
        train(dataLoader_train, netmodel, optimizer, epoch, logger_train, exp_args)
        print ('===========>   testing    <===========')
        loss = test(dataLoader_test, netmodel, optimizer, epoch, logger_test, exp_args)
        print ("loss: ", loss, minLoss)
        is_best = False
        if loss < minLoss:
            minLoss = loss
            is_best = True
        
        save_checkpoint({
            'epoch': epoch+1,
            'minLoss': minLoss,
            'state_dict': netmodel.state_dict(),
            'optimizer' : optimizer.state_dict(),
            }, is_best, exp_args.model_root)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training code')
    parser.add_argument('--model', default='PortraitNet', type=str, 
                        help='<model> should in [PortraitNet, ENet, BiSeNet]')
    parser.add_argument('--config_path', 
                        default='/home/dongx12/PortraitNet/config/model_mobilenetv2_without_auxiliary_losses.yaml', 
                        type=str, help='the config path of the model')
    
    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers')
    parser.add_argument('--batchsize', default=64, type=int, help='mini-batch size')
    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weightdecay', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--printfreq', default=100, type=int, help='print frequency')
    parser.add_argument('--savefreq', default=1000, type=int, help='save frequency')
    parser.add_argument('--resume', default=False, type=bool, help='resume')
    args = parser.parse_args()
    
    main(args)