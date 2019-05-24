import os
import cv2
import math
import random
import scipy
import json
import copy
import base64
import zlib
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from PIL import Image, ImageEnhance, ImageOps, ImageFile  

import sys
sys.path.insert(0, '/home/dongx12/Data/cocoapi/PythonAPI/')
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

# global parameter
set_ratio = 0.5

def load_json(fileName):
    with open(fileName,'r') as data_file:
        anno = json.load(data_file)
    return anno

def mask_to_bbox(mask):
    site = np.where(mask>0)
    bbox = [np.min(site[1]), np.min(site[0]), np.max(site[1]), np.max(site[0])]
    return bbox

# ===================== generate edge for input image =====================
def show_edge(mask_ori):
    mask = mask_ori.copy()
    # find countours: img must be binary
    myImg = np.zeros((mask.shape[0], mask.shape[1]), np.uint8)
    ret, binary = cv2.threshold(np.uint8(mask)*255, 127, 255, cv2.THRESH_BINARY)
    img, countours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # RETR_EXTERNAL
    '''
    cv2.drawContours(myImg, countours, -1, 1, 10)
    diff = mask + myImg
    diff[diff < 2] = 0
    diff[diff == 2] = 1
    return diff   
    '''
    cv2.drawContours(myImg, countours, -1, 1, 4)
    return myImg

# ===================== load mask =====================
def annToRLE(anno, height, width):
    """
    Convert annotation which can be polygons, uncompressed RLE to RLE.
    :return: binary mask (numpy 2D array)
    """
    segm = anno['segmentation']
    if isinstance(segm, list):
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(segm, height, width)
        rle = maskUtils.merge(rles)
    elif isinstance(segm['counts'], list):
        # uncompressed RLE
        rle = maskUtils.frPyObjects(segm, height, width)
    else:
        # rle
        rle = ann['segmentation']
    return rle

def annToMask(anno, height, width):
    """
    Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
    :return: binary mask (numpy 2D array)
    """
    rle = annToRLE(anno, height, width)
    mask = maskUtils.decode(rle)
    return mask

def base64_2_mask(s):
    z = zlib.decompress(base64.b64decode(s))
    n = np.fromstring(z, np.uint8)
    mask = cv2.imdecode(n, cv2.IMREAD_UNCHANGED)[:, :, 3].astype(bool)
    return mask

def mask_2_base64(mask):
    img_pil = Image.fromarray(np.array(mask, dtype=np.uint8))
    img_pil.putpalette([0,0,0,255,255,255])
    bytes_io = io.BytesIO()
    img_pil.save(bytes_io, format='PNG', transparency=0, optimize=0)
    bytes = bytes_io.getvalue()
    return base64.b64encode(zlib.compress(bytes)).decode('utf-8')

# ===================== deformable data augmentation for input image =====================
def flip_data(width, keypoint_ori):
    keypoint = copy.deepcopy(keypoint_ori)
    for i in xrange(len(keypoint)/3):
        keypoint[3*i] = width - 1 - keypoint[3*i]
    right = [2,4, 6,8,10, 12,14,16]
    left  = [1,3, 5,7,9,  11,13,15]
    
    for i in xrange(len(left)):
        temp = copy.deepcopy(keypoint[3*right[i]:3*(right[i]+1)]) 
        keypoint[3*right[i]:3*(right[i]+1)] = keypoint[3*left[i]:3*(left[i]+1)]
        keypoint[3*left[i]:3*(left[i]+1)] = temp
    return keypoint

def data_aug_flip(image, mask):
    if random.random()<set_ratio:
        return image, mask, False
    return image[:,::-1,:], mask[:,::-1], True

def aug_matrix(img_w, img_h, bbox, w, h, angle_range=(-45, 45), scale_range=(0.5, 1.5), offset=40):
    ''' 
    first Translation, then rotate, final scale.
        [sx, 0, 0]       [cos(theta), -sin(theta), 0]       [1, 0, dx]       [x]
        [0, sy, 0] (dot) [sin(theta),  cos(theta), 0] (dot) [0, 1, dy] (dot) [y]
        [0,  0, 1]       [         0,           0, 1]       [0, 0,  1]       [1]
    '''
    ratio = 1.0*(bbox[2]-bbox[0])*(bbox[3]-bbox[1])/(img_w*img_h)
    x_offset = (random.random()-0.5) * 2 * offset
    y_offset = (random.random()-0.5) * 2 * offset
    dx = (w-(bbox[2]+bbox[0]))/2.0 
    dy = (h-(bbox[3]+bbox[1]))/2.0
    
    matrix_trans = np.array([[1.0, 0, dx],
                             [0, 1.0, dy],
                             [0, 0,   1.0]])

    angle = random.random()*(angle_range[1]-angle_range[0])+angle_range[0]
    scale = random.random()*(scale_range[1]-scale_range[0])+scale_range[0]
    scale *= np.mean([float(w)/(bbox[2]-bbox[0]), float(h)/(bbox[3]-bbox[1])])
    alpha = scale * math.cos(angle/180.0*math.pi)
    beta = scale * math.sin(angle/180.0*math.pi)

    centerx = w/2.0 + x_offset
    centery = h/2.0 + y_offset
    H = np.array([[alpha, beta, (1-alpha)*centerx-beta*centery], 
                  [-beta, alpha, beta*centerx+(1-alpha)*centery],
                  [0,         0,                            1.0]])

    H = H.dot(matrix_trans)[0:2, :]
    return H  

# ===================== texture data augmentation for input image =====================
def data_aug_light(image):
    if random.random()<set_ratio:
        return image
    value = random.randint(-30, 30)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image = np.array(hsv_image, dtype=np.float32)
    hsv_image[:,:,2] += value
    hsv_image[hsv_image>255] = 255
    hsv_image[hsv_image<0] = 0
    hsv_image = np.array(hsv_image, dtype=np.uint8)
    image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    return image    

def data_aug_blur(image):
    if random.random()<set_ratio:
        return image
    
    select = random.random()
    if select < 0.3:
        kernalsize = random.choice([3,5])
        image = cv2.GaussianBlur(image, (kernalsize,kernalsize),0)
    elif select < 0.6:
        kernalsize = random.choice([3,5])
        image = cv2.medianBlur(image, kernalsize)
    else:
        kernalsize = random.choice([3,5])
        image = cv2.blur(image, (kernalsize,kernalsize))
    return image

def data_aug_color(image):  
    if random.random()<set_ratio:
        return image
    random_factor = np.random.randint(4, 17) / 10. 
    color_image = ImageEnhance.Color(image).enhance(random_factor) 
    random_factor = np.random.randint(4, 17) / 10. 
    brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)
    random_factor = np.random.randint(6, 15) / 10. 
    contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)
    random_factor = np.random.randint(8, 13) / 10.
    return ImageEnhance.Sharpness(contrast_image).enhance(random_factor)

def data_aug_noise(image):
    if random.random()<set_ratio:
        return image
    mu = 0
    sigma = random.random()*10.0
    image = np.array(image, dtype=np.float32)
    image += np.random.normal(mu, sigma, image.shape)
    image[image>255] = 255
    image[image<0] = 0
    return image

# ===================== normalization for input image =====================
def padding(img_ori, mask_ori, size=224, padding_color=128):
    height = img_ori.shape[0]
    width = img_ori.shape[1]
    
    img = np.zeros((max(height, width), max(height, width), 3)) + padding_color
    mask = np.zeros((max(height, width), max(height, width)))
    
    if (height > width):
        padding = int((height-width)/2)
        img[:, padding:padding+width, :] = img_ori
        mask[:, padding:padding+width] = mask_ori
    else:
        padding = int((width-height)/2)
        img[padding:padding+height, :, :] = img_ori
        mask[padding:padding+height, :] = mask_ori
        
    img = np.uint8(img)
    mask = np.uint8(mask)
    
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_CUBIC)
    
    return np.array(img, dtype=np.float32),  np.array(mask, dtype=np.float32)

def Normalize_Img(imgOri, scale, mean, val):
    img = np.array(imgOri.copy(), np.float32)/scale
    if len(img.shape) == 4:
        for j in range(img.shape[0]):
            for i in range(len(mean)):
                img[j,:,:,i] = (img[j,:,:,i]-mean[i])/val[i]
        return img
    else:
        for i in range(len(mean)):
            img[:,:,i] = (img[:,:,i]-mean[i])/val[i]
        return img

def Anti_Normalize_Img(imgOri, scale, mean, val):
    img = np.array(imgOri.copy(), np.float32)
    if len(img.shape) == 4:
        for j in range(img.shape[0]):
            for i in range(len(mean)):
                img[j,:,:,i] = img[j,:,:,i]*val[i]+mean[i]
        return np.array(img*scale, np.uint8)
    else:
        for i in range(len(mean)):
            img[:,:,i] = img[:,:,i]*val[i]+mean[i]
        return np.array(img*scale, np.uint8)
    
# ===================== generate prior channel for input image =====================
def data_motion_blur(image, mask):
    if random.random()<set_ratio:
        return image, mask
    
    degree = random.randint(5, 30)
    angle = random.randint(0, 360)
    
    M = cv2.getRotationMatrix2D((degree/2, degree/2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
    motion_blur_kernel = motion_blur_kernel/degree
    
    img_blurred = cv2.filter2D(image, -1, motion_blur_kernel)
    mask_blurred = cv2.filter2D(mask, -1, motion_blur_kernel)
    
    cv2.normalize(img_blurred, img_blurred, 0, 255, cv2.NORM_MINMAX)
    cv2.normalize(mask_blurred, mask_blurred, 0, 1, cv2.NORM_MINMAX)
    return img_blurred, mask_blurred
    
def data_motion_blur_prior(prior):
    if random.random()<set_ratio:
        return prior
    
    degree = random.randint(5, 30)
    angle = random.randint(0, 360)
    
    M = cv2.getRotationMatrix2D((degree/2, degree/2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
    motion_blur_kernel = motion_blur_kernel/degree
    
    prior_blurred = cv2.filter2D(prior, -1, motion_blur_kernel)
    return prior_blurred  
    
def data_Affine(image, mask, height, width, ratio=0.05):
    if random.random()<set_ratio:
        return image, mask
    bias = np.random.randint(-int(height*ratio),int(width*ratio), 12)
    pts1 = np.float32([[0+bias[0], 0+bias[1]], [width+bias[2], 0+bias[3]], [0+bias[4], height+bias[5]]])
    pts2 = np.float32([[0+bias[6], 0+bias[7]], [width+bias[8], 0+bias[9]], [0+bias[10], height+bias[11]]])
    M = cv2.getAffineTransform(pts1, pts2)
    img_affine = cv2.warpAffine(image, M, (width, height))
    mask_affine = cv2.warpAffine(mask, M, (width, height))
    return img_affine, mask_affine

def data_Affine_prior(prior, height, width, ratio=0.05):
    if random.random()<set_ratio:
        return prior
    bias = np.random.randint(-int(height*ratio),int(width*ratio), 12)
    pts1 = np.float32([[0+bias[0], 0+bias[1]], [width+bias[2], 0+bias[3]], [0+bias[4], height+bias[5]]])
    pts2 = np.float32([[0+bias[6], 0+bias[7]], [width+bias[8], 0+bias[9]], [0+bias[10], height+bias[11]]])
    M = cv2.getAffineTransform(pts1, pts2)
    prior_affine = cv2.warpAffine(prior, M, (width, height))
    return prior_affine
    
def data_Perspective(image, mask, height, width, ratio=0.05):
    if random.random()<set_ratio:
        return image, mask
    bias = np.random.randint(-int(height*ratio),int(width*ratio), 16)
    pts1 = np.float32([[0+bias[0],0+bias[1]], [height+bias[2],0+bias[3]], 
                       [0+bias[4],width+bias[5]], [height+bias[6], width+bias[7]]])
    pts2 = np.float32([[0+bias[8],0+bias[9]], [height+bias[10],0+bias[11]], 
                       [0+bias[12],width+bias[13]], [height+bias[14], width+bias[15]]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    img_perspective = cv2.warpPerspective(image, M, (width, height))
    mask_perspective = cv2.warpPerspective(mask, M, (width, height))
    return img_perspective, mask_perspective

def data_Perspective_prior(prior, height, width, ratio=0.05):
    if random.random()<set_ratio:
        return prior
    bias = np.random.randint(-int(height*ratio),int(width*ratio), 16)
    pts1 = np.float32([[0+bias[0],0+bias[1]], [height+bias[2],0+bias[3]], 
                       [0+bias[4],width+bias[5]], [height+bias[6], width+bias[7]]])
    pts2 = np.float32([[0+bias[8],0+bias[9]], [height+bias[10],0+bias[11]], 
                       [0+bias[12],width+bias[13]], [height+bias[14], width+bias[15]]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    prior_perspective = cv2.warpPerspective(prior, M, (width, height))
    return prior_perspective

def data_ThinPlateSpline(image, mask, height, width, ratio=0.05):
    if random.random()<set_ratio:
        return image, mask
    bias = np.random.randint(-int(height*ratio),int(width*ratio), 16)
    tps = cv2.createThinPlateSplineShapeTransformer()
    sshape = np.array([[0+bias[0],0+bias[1]], [height+bias[2],0+bias[3]], 
                       [0+bias[4],width+bias[5]], [height+bias[6], width+bias[7]]], np.float32)
    tshape = np.array([[0+bias[8],0+bias[9]], [height+bias[10],0+bias[11]], 
                       [0+bias[12],width+bias[13]], [height+bias[14], width+bias[15]]], np.float32)
    sshape = sshape.reshape(1,-1,2)
    tshape = tshape.reshape(1,-1,2)
    matches = list()
    matches.append(cv2.DMatch(0,0,0))
    matches.append(cv2.DMatch(1,1,0))
    matches.append(cv2.DMatch(2,2,0))
    matches.append(cv2.DMatch(3,3,0))
    
    tps.estimateTransformation(tshape, sshape, matches)
    res = tps.warpImage(image)
    res_mask = tps.warpImage(mask)
    return res, res_mask   

def data_ThinPlateSpline_prior(prior, height, width, ratio=0.05):
    if random.random()<set_ratio:
        return prior
    bias = np.random.randint(-int(height*ratio),int(width*ratio), 16)
    tps = cv2.createThinPlateSplineShapeTransformer()
    sshape = np.array([[0+bias[0],0+bias[1]], [height+bias[2],0+bias[3]], 
                       [0+bias[4],width+bias[5]], [height+bias[6], width+bias[7]]], np.float32)
    tshape = np.array([[0+bias[8],0+bias[9]], [height+bias[10],0+bias[11]], 
                       [0+bias[12],width+bias[13]], [height+bias[14], width+bias[15]]], np.float32)
    sshape = sshape.reshape(1,-1,2)
    tshape = tshape.reshape(1,-1,2)
    matches = list()
    matches.append(cv2.DMatch(0,0,0))
    matches.append(cv2.DMatch(1,1,0))
    matches.append(cv2.DMatch(2,2,0))
    matches.append(cv2.DMatch(3,3,0))
    
    tps.estimateTransformation(tshape, sshape, matches)
    prior = tps.warpImage(prior)
    return prior