import os
import sys
import numpy as np
import cbf
import re
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage import morphology
from skimage.measure import label, regionprops
from skimage.draw import polygon_perimeter
import logging
import time


def get_gts(mannualf = 'data/Files-31Aug2021/best-but-rings.manual.txt'):
    gts = []
    with open(mannualf, "r") as f:
        f.readline()
        for r in f:
            position = re.split(',| |\t', r)
            if position:
                    gts += [ [int(float(position[0])),int(float(position[1]))] ]
    return np.asarray(gts)


def mask_possprocess(mask,th = 0.4):
    mask_poseprocessed = (mask>th).astype(np.int32)
    #mask_poseprocessed[:,:20]=0
    #mask_poseprocessed[:,-20:]=0
    #mask_poseprocessed[:,:,:20]=0
    #mask_poseprocessed[:,:,-20:]=0
    for i,mask in enumerate(mask_poseprocessed):
        # mask_poseprocessed[i] = morphology.binary_opening(mask, morphology.diamond(3)).astype(np.int32)
        mask_poseprocessed[i] = morphology.binary_opening(mask, morphology.disk(3)).astype(np.int32)
    return mask_poseprocessed
    

def detection_metrix(gtmask, mask, th=0.5):
    # tp,ngt,npredic = 0., 0., 0.
    tp, fp, tn, fn = 0., 0., 0., 0.
    spot_num = 0
    for i in range(len(gtmask)):
        labeled_mask = label(mask[i, :, :])
        props = regionprops(labeled_mask)
        for x in props:
            if gtmask.ndim == 4:
                if gtmask[i, :, :, 0][int(x.centroid[0]), int(x.centroid[1])] < 1:
                    fp += 1
                else:
                    tp += 1
            else:
                if gtmask[i][int(x.centroid[0]), int(x.centroid[1])] < 1:
                    fp += 1
                else:
                    tp += 1
        if gtmask.ndim == 4:
            labeled_gtmask = label(gtmask[i, :, :, 0])
        else:
            labeled_gtmask = label(gtmask[i])
        props = regionprops(labeled_gtmask)
        spot_num += len(props)
        for x in props:
            if mask[i][int(float(x.centroid[0])), int(float(x.centroid[1]))] < 1:
                fn += 1
    npredic = tp + fp
    ngt = tp + fn
    if ngt == 0:
        ngt = spot_num
    return tp,ngt,npredic


def load_data(imf = '../data/Files-21Oct2021/mx308820-2_hetB_B-2_026.cbf',lbsf = '../data/Files-21Oct2021/mx308820-2_hetB_B-2_026_manual.txt',label_radius = 10):
    #lbsf = lbsf.replace('manual',label_suffix)
    image_gray = cbf.read(imf).data
    if image_gray.max()>60000:
        image_gray[image_gray==image_gray.max()] = 0
    image_gray[image_gray>255] = 255
    image_gray = image_gray/255
    im = []
    lbs = []
    labels = np.zeros_like(image_gray)
    gts = get_gts(lbsf)
    for position in gts:
        if ('albula' in lbsf) or ('dozor' in lbsf) or ('dials' in lbsf):# 
                labels[int(float(position[1])),int(float(position[0]))] = 1
        else:
                labels[int(float(position[0])),int(float(position[1]))] = 1
    labels = morphology.binary_dilation(labels, morphology.diamond(label_radius)).astype(np.uint8)
    w,h = image_gray.shape
    print(w,h)
    for xi,x in enumerate(range(w%512//2,w//512*512+w%512//2,512)):
            for yi,y in enumerate(range(h%512//2,h//512*512+h%512//2,512)):
                lbs_patch = labels[x:x+512,y:y+512]
                if lbs_patch.max()<1:
                    continue
                lbs.append(lbs_patch)
                image_patch = image_gray[x:x+512,y:y+512]
                im.append(image_patch)
    im = np.asarray(im).reshape([-1,512,512,1])
    lbs = np.asarray(lbs).reshape([-1,512,512,1])
    return im,lbs,gts


def load_data_quicknpy(datadir = '../data/Files_before31Dec2021/'):
    return np.load(datadir+'ims_manual.npy'), np.load(datadir+'lbs_manual.npy')


def Prepare_logger(log_name, print_console=True):
    logger = logging.getLogger(__name__)
    logger.propagate = False
    logger.setLevel(logging.INFO)
    if print_console:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s %(message)s')
        handler.setFormatter(formatter)
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)
    date = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    logfile = log_name + date + '.log'
    file_handler = logging.FileHandler(logfile, mode='w')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

