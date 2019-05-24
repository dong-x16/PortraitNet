import torch
import torch.utils.data as data

import os
import cv2
import sys
import numpy as np
import math
import random
import scipy
from scipy.ndimage.filters import gaussian_filter
from easydict import EasyDict as edict

import json
import time
import copy
from PIL import Image
# from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True

from data_aug import data_aug_blur, data_aug_color, data_aug_noise, data_aug_light
from data_aug import data_aug_flip, flip_data, aug_matrix
from data_aug import show_edge, mask_to_bbox, load_json
from data_aug import base64_2_mask, mask_2_base64, padding, Normalize_Img, Anti_Normalize_Img
from data_aug import data_motion_blur, data_Affine, data_Perspective, data_ThinPlateSpline
from data_aug import data_motion_blur_prior, data_Affine_prior, data_Perspective_prior, data_ThinPlateSpline_prior

class PortraitSeg(data.Dataset): 
    def __init__(self, ImageRoot, AnnoRoot, ImgIds_Train, ImgIds_Test, exp_args):
        self.ImageRoot = ImageRoot
        self.AnnoRoot = AnnoRoot
        self.istrain = exp_args.istrain
        self.stability = exp_args.stability
        self.addEdge = exp_args.addEdge
        
        self.video = exp_args.video
        self.prior_prob = exp_args.prior_prob
        
        self.task = exp_args.task
        self.dataset = exp_args.dataset
        self.input_height = exp_args.input_height
        self.input_width = exp_args.input_width
        
        self.padding_color = exp_args.padding_color
        self.img_scale = exp_args.img_scale
        self.img_mean = exp_args.img_mean # BGR order
        self.img_val = exp_args.img_val # BGR order
        
        if self.istrain == True:
            file_object = open(ImgIds_Train, 'r')
        elif self.istrain == False:
            file_object = open(ImgIds_Test, 'r')
            
        try:
            self.imgIds = file_object.readlines()
            if self.dataset == "MscocoBackground" and self.istrain == True:
                self.imgIds = self.imgIds[:5000]
                
            if self.dataset == "ATR" and self.istrain == True:
                self.imgIds = self.imgIds[:5000]
            
            # if self.istrain == False:
            #     self.imgIds = self.imgIds[:100]
                
        finally:
             file_object.close()
        pass
            
        
    def __getitem__(self, index):
        '''
        An item is an image. Which may contains more than one person.
        '''
        img = None
        mask = None
        bbox = None
        H = None
        
        if self.dataset == "supervisely":
            # basic info
            img_path = os.path.join(self.ImageRoot, self.imgIds[index].strip())
            img_name = img_path[img_path.rfind('/')+1:]
            img = cv2.imread(img_path)
            
            # load mask
            annopath = img_path.replace('/img/', '/ann/')
            annopath = annopath[:annopath.find('.')]+'.json'
            ann = load_json(annopath)
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
            for i in range(len(ann['objects'])):
                mask_temp = np.zeros((img.shape[0], img.shape[1]))
                if ann['objects'][i]['classTitle'] == 'person_poly':
                    points = np.array(ann['objects'][i]['points']['exterior'])
                    if len(points) > 0:
                        cv2.fillPoly(mask_temp, [points], 1)
                        points = np.array(ann['objects'][i]['points']['interior'])
                        for p in points:
                            cv2.fillPoly(mask_temp, [np.array(p)], 0)
                elif ann['objects'][i]['classTitle'] == 'neutral':
                    points = np.array(ann['objects'][i]['points']['exterior'])
                    if len(points) > 0:
                        cv2.fillPoly(mask_temp, [points], 1)
                        points = np.array(ann['objects'][i]['points']['interior'])
                        for p in points:
                            cv2.fillPoly(mask_temp, [np.array(p)], 0)
                elif ann['objects'][i]['classTitle'] == 'person_bmp':
                    data = np.array(ann['objects'][i]['bitmap']['data'])
                    if data.size > 0:
                        mask_ = base64_2_mask(data)
                        origin = ann['objects'][i]['bitmap']['origin']
                        mask_temp[origin[1]:origin[1]+mask_.shape[0], origin[0]:origin[0]+mask_.shape[1]] = mask_
                mask[mask_temp>0] = 1
            
            height, width, channel = img.shape
            # bbox = mask_to_bbox(mask)
            bbox = [0, 0, width-1, height-1]
            
            H = aug_matrix(width, height, bbox, self.input_width, self.input_height,
                       angle_range=(-45, 45), scale_range=(0.5, 1.5), offset=self.input_height/4)
            
        elif self.dataset in ["supervisely_face_easy", "supervisely_face_difficult"]:
            # basic info
            img_path = os.path.join(self.ImageRoot, self.imgIds[index].strip())
            img_name = img_path[img_path.rfind('/')+1:]
            img = cv2.imread(img_path)
            # img = cv2.imread(img_path.replace('/img/', '/imgAug/'))
            
            # load mask
            annopath = img_path.replace('/img/', '/ann/')
            # annopath = img_path.replace('/img/', '/maskAug/')
            
            mask = cv2.imread(annopath, 0) # origin mask = 255
            mask[mask>0] = 1
            
            height, width, channel = img.shape
            # bbox = mask_to_bbox(mask)
            bbox = [0, 0, width-1, height-1]
            H = aug_matrix(width, height, bbox, self.input_width, self.input_height,
                       angle_range=(-45, 45), scale_range=(0.5, 1.5), offset=self.input_height/4)
            
        elif self.dataset in ["flickr", "eg1800", "liveshow"]:
            # basic info
            img_id = self.imgIds[index].strip()
            img_path = os.path.join(self.ImageRoot, img_id)
            img = cv2.imread(img_path)
            # img = cv2.imread(img_path.replace('Images', 'ImagesAug'))
            img_name = img_path[img_path.rfind('/')+1:]
            
            # load mask
            annopath = os.path.join(self.AnnoRoot, img_id.replace('.jpg', '.png'))
            mask = cv2.imread(annopath, 0)
            # mask = cv2.imread(annopath.replace('Labels', 'LabelsAug'), 0)
            mask[mask>1] = 0
            
            height, width, channel = img.shape
            bbox = [0, 0, width-1, height-1]
            H = aug_matrix(width, height, bbox, self.input_width, self.input_height,
                       angle_range=(-45, 45), scale_range=(0.5, 1.5), offset=self.input_height/4)
        
        elif self.dataset == "ATR":
            # basic info
            img_id = self.imgIds[index].strip()
            img_path = os.path.join(self.ImageRoot, img_id)
            img = cv2.imread(img_path)
            img_name = img_path[img_path.rfind('/')+1:]
            
            # load mask
            annopath = os.path.join(self.AnnoRoot, img_id.replace('.jpg', '.png'))
            mask = cv2.imread(annopath, 0) 
            mask[mask>1] = 1
            
            height, width, channel = img.shape
            bbox = [0, 0, width-1, height-1]
            H = aug_matrix(width, height, bbox, self.input_width, self.input_height,
                       angle_range=(-45, 45), scale_range=(0.5, 1.5), offset=self.input_height/4)
        
        elif self.dataset == "MscocoBackground":
            # basic info
            img_path = self.imgIds[index].strip()
            img_path = os.path.join(self.ImageRoot, img_path)
            img = cv2.imread(img_path)
            height, width, channel = img.shape
            mask = np.zeros((height, width))
            
            bbox = [0, 0, width-1, height-1]
            H = aug_matrix(width, height, bbox, self.input_width, self.input_height,
                       angle_range=(-45, 45), scale_range=(1.5, 2.0), offset=self.input_height/4)
            
        use_float_mask = False # use original 0/1 mask as groundtruth
        
        # data augument: first align center to center of dst size. then rotate and scale
        if self.istrain == False:
            img_aug_ori, mask_aug_ori = padding(img, mask, size=self.input_width, padding_color=self.padding_color)
            
            # ===========add new channel for video stability============
            input_norm = Normalize_Img(img_aug_ori, scale=self.img_scale, mean=self.img_mean, val=self.img_val)
            if self.video == True:
                prior = np.zeros((self.input_height, self.input_width, 1))
                input_norm = np.c_[input_norm, prior]
            input = np.transpose(input_norm, (2, 0, 1))
            input_ori = copy.deepcopy(input)
        else:
            img_aug = cv2.warpAffine(np.uint8(img), H, (self.input_width, self.input_height), 
                                     flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, 
                                     borderValue=(self.padding_color, self.padding_color, self.padding_color)) 
            mask_aug = cv2.warpAffine(np.uint8(mask), H, (self.input_width, self.input_height), 
                                      flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
            img_aug_ori, mask_aug_ori, aug_flag = data_aug_flip(img_aug, mask_aug)
            prior = np.zeros((self.input_height, self.input_width, 1))
            
            # ======== add new channel for video stability =========
            if self.video == True and self.prior_prob >= random.random(): # add new augmentation
                prior[:,:,0] = mask_aug_ori.copy()
                prior = np.array(prior, dtype=np.float)
                
                if random.random() >= 0.5:
                    # modify image + mask, use groundtruth as prior
                    img_aug_ori = np.array(img_aug_ori)
                    mask_aug_ori = np.array(mask_aug_ori, dtype=np.float)
                    img_aug_ori, mask_aug_ori = data_motion_blur(img_aug_ori, mask_aug_ori)
                    img_aug_ori, mask_aug_ori = data_Affine(img_aug_ori, mask_aug_ori, self.input_height, self.input_width, ratio=0.05)
                    img_aug_ori, mask_aug_ori = data_Perspective(img_aug_ori, mask_aug_ori, self.input_height, self.input_width, ratio=0.05)
                    img_aug_ori, mask_aug_ori = data_ThinPlateSpline(img_aug_ori, mask_aug_ori, self.input_height, self.input_width, ratio=0.05)
                    use_float_mask = True
                else:
                    # modify prior, don't change image + mask
                    prior = data_motion_blur_prior(prior)
                    prior = data_Affine_prior(prior, self.input_height, self.input_width, ratio=0.05)
                    prior = data_Perspective_prior(prior, self.input_height, self.input_width, ratio=0.05)
                    prior = data_ThinPlateSpline_prior(prior, self.input_height, self.input_width, ratio=0.05)
                    prior = prior.reshape(self.input_height, self.input_width, 1)
                
            # add augmentation
            img_aug = Image.fromarray(cv2.cvtColor(img_aug_ori, cv2.COLOR_BGR2RGB))  
            img_aug = data_aug_color(img_aug)
            img_aug = np.asarray(img_aug)
            # img_aug = data_aug_light(img_aug)
            img_aug = data_aug_blur(img_aug)
            img_aug = data_aug_noise(img_aug)
            img_aug = np.float32(img_aug[:,:,::-1]) # BGR, like cv2.imread
            
            input_norm = Normalize_Img(img_aug, scale=self.img_scale, mean=self.img_mean, val=self.img_val)
            input_ori_norm = Normalize_Img(img_aug_ori, scale=self.img_scale, mean=self.img_mean, val=self.img_val)
                
            if self.video == True:
                input_norm = np.c_[input_norm, prior]
                input_ori_norm = np.c_[input_ori_norm, prior]
            
            input = np.transpose(input_norm, (2, 0, 1))
            input_ori = np.transpose(input_ori_norm, (2, 0, 1))
            
        if 'seg' in self.task:
            if use_float_mask == True:
                output_mask = cv2.resize(mask_aug_ori, (self.input_width, self.input_height), interpolation=cv2.INTER_NEAREST)
                cv2.normalize(output_mask, output_mask, 0, 1, cv2.NORM_MINMAX)
                output_mask[output_mask>=0.5] = 1
                output_mask[output_mask<0.5] = 0
            else:
                output_mask = cv2.resize(np.uint8(mask_aug_ori), (self.input_width, self.input_height), interpolation=cv2.INTER_NEAREST)
                
                # add mask blur
                output_mask = np.uint8(cv2.blur(output_mask, (5,5)))
                output_mask[output_mask>=0.5] = 1
                output_mask[output_mask<0.5] = 0
        else:
            output_mask = np.zeros((self.input_height, self.input_width), dtype=np.uint8) + 255
        
        if self.task == 'seg':
            edge = show_edge(output_mask)
            # edge_blur = np.uint8(cv2.blur(edge, (5,5)))/255.0
            return input_ori, input, edge, output_mask
            
    def __len__(self):
        return len(self.imgIds)