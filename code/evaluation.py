import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from skimage.measure import label, regionprops
from skimage import morphology, draw
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from data_preprocess import get_gts, load_data_full_img
import copy
import cv2
from util import *
from visualization import parse_header, fix_header
from tqdm import tqdm
import os
import glob
import cbf
import json


ring_img_dir = './ice_ring_correction/corrected_old_ring_test/'


def model_evaluation(labels, predicts, patch_id, patch_pos, full_path_dict, label_radius=10, mask_ring=False, print_result=True):
    image_id = list(set(patch_id))
    ring_full_dict = {}
    new_label_full_dict = {}
    images_full = None
    for im_id in image_id:
        imf = full_path_dict[im_id]
        lbsf = imf[:-3] + 'manual.txt'
        # parse header
        out = parse_header(imf)
        if out is None:
            radius, center, _ = fix_header()
        else:
            radius, center, _ = out
        ring_full_dict[im_id] = [radius, center]
        images_full, labels_full = load_data_full_img(imf, lbsf, label_radius)
        images_full = (images_full - np.min(images_full)) / (np.max(images_full) - np.min(images_full))
        new_label_full_dict[im_id] = labels_full
        ring_img_full_path = ring_img_dir + im_id[:-4] + '_ring.jpg'
        ring_img = cv2.imread(ring_img_full_path, cv2.IMREAD_GRAYSCALE)
        if labels_full.shape != ring_img.shape:
            print(im_id, labels_full.shape, ring_img.shape)
    w, h = images_full.shape
    new_predict_full_dict = {}
    for im_id in image_id:
        new_predict_full_dict[im_id] = np.zeros((w, h))

    for idx in range(len(patch_pos)):
        im_id = patch_id[idx]
        pos_x, pos_y = patch_pos[idx]
        patchx1, patchx2 = w % 512 // 2 + pos_x * 512, w % 512 // 2 + (pos_x + 1) * 512
        patchy1, patchy2 = h % 512 // 2 + pos_y * 512, h % 512 // 2 + (pos_y + 1) * 512
        patchx1 = int(patchx1)
        patchx2 = int(patchx2)
        patchy1 = int(patchy1)
        patchy2 = int(patchy2)
        new_predict_full_dict[im_id][patchx1:patchx2, patchy1:patchy2] += predicts[idx]

    for i, im_id in enumerate(image_id):
        pred = new_predict_full_dict[im_id]
        pred_labels = label(pred)  # label different regions
        props = regionprops(pred_labels)
        # split large region
        for x in props:
            if (x.area > 260) and ((x.bbox[2] - x.bbox[0]) > 20 or (x.bbox[3] - x.bbox[1]) > 20):
                pred_bbox = pred[x.bbox[0]:x.bbox[2], x.bbox[1]:x.bbox[3]]
                distance = ndi.distance_transform_edt(pred_bbox)
                # coords = peak_local_max(distance, footprint=np.ones((10, 10)), labels=res)
                coords = peak_local_max(distance, min_distance=7, labels=pred_bbox)
                mask = np.zeros(distance.shape, dtype=bool)
                mask[tuple(coords.T)] = True
                markers, _ = ndi.label(mask)
                labels_tmp = watershed(-distance, markers, mask=pred_bbox)
                new_predict_full_dict[im_id][x.bbox[0]:x.bbox[2], x.bbox[1]:x.bbox[3]] = labels_tmp
        # remove small region
        pred = new_predict_full_dict[im_id]
        pred_labels = label(pred)  # label different regions
        props = regionprops(pred_labels)
        for x in props:
            if (x.area < 40) or ((x.bbox[2] - x.bbox[0]) < 5) or ((x.bbox[3] - x.bbox[1]) < 5):
                new_predict_full_dict[im_id][tuple(x.coords.T)] = 0.
        if mask_ring:
            radius, center = ring_full_dict[im_id]
            labels = label(pred)  # label different regions
            props = regionprops(labels)
            for x in props:
                distance_to_center = x.centroid - center
                distance_to_center = np.sqrt(np.sum(distance_to_center ** 2))
                for r in radius:
                    if abs(distance_to_center - r) < 3:
                        new_predict_full_dict[im_id][tuple(x.coords.T)] = 0.
                if (distance_to_center - 60.) < 0:
                    new_predict_full_dict[im_id][tuple(x.coords.T)] = 0.

        if mask_ring:
            ring_img_full_path = ring_img_dir + im_id[:-4] + '_ring.jpg'
            ring_img = cv2.imread(ring_img_full_path, cv2.IMREAD_GRAYSCALE)
            if ring_img is None:
                print(ring_img_full_path)
                continue
            ring_img = ring_img / 255
            ring_img[ring_img > 0.5] = 1  # artifacts caused by detector?
            ring_img[ring_img < 0.5] = 0  # grids are -1, artifacts are -2 values?

            labels = label(pred)  # label different regions
            props = regionprops(labels)
            for x in props:
                distance_to_center = x.centroid - center
                distance_to_center = np.sqrt(np.sum(distance_to_center ** 2))
                if (distance_to_center - 60.) < 0:
                    new_predict_full_dict[im_id][tuple(x.coords.T)] = 0.
                try:
                    if ring_img[int(x.centroid[0]), int(x.centroid[1])] > 0:
                        new_predict_full_dict[im_id][tuple(x.coords.T)] = 0.
                except:
                    print(x.centroid, ring_img.shape)
                    continue

    metrics_test = detection_metrix(np.array(list(new_label_full_dict.values())),
                                    np.array(list(new_predict_full_dict.values())))
    try:
        prec_avg, rec_avg = metrics_test[0] / metrics_test[2], metrics_test[0] / metrics_test[1]
    except:
        print(metrics_test)
    f1_avg = 2 * prec_avg * rec_avg / (prec_avg + rec_avg)
    
    # # image level
    # prec_avg = []
    # rec_avg = []
    # f1_avg = []
    # label_gt = np.array(list(new_label_full_dict.values()))
    # image_pred = np.array(list(new_predict_full_dict.values()))
    # for i in range(len(label_gt)):
    #     metrics_test = detection_metrix(label_gt[i:(i+1)], image_pred[i:(i+1)])
    #     if metrics_test[2] == 0:
    #         prec = 0
    #         if metrics_test[1] == 0:
    #             rec = 0
    #         else:
    #             rec = metrics_test[0] / metrics_test[1]
    #     else:
    #         prec, rec = metrics_test[0] / metrics_test[2], metrics_test[0] / metrics_test[1]
    #     if (prec + rec) == 0:
    #         f1 = 0
    #     else:
    #         f1 = 2 * prec * rec / (prec + rec)
    #     prec_avg.append(prec)
    #     rec_avg.append(rec)
    #     f1_avg.append(f1)
    # prec_avg = sum(prec_avg) / len(prec_avg)
    # rec_avg = sum(rec_avg) / len(rec_avg)
    # f1_avg = sum(f1_avg) / len(f1_avg)
    
    if print_result:
        print(f'average precision {prec_avg:.5f}, average recall {rec_avg:.5f}, average f1 score {f1_avg:.5f}')
    return prec_avg, rec_avg, f1_avg


def software_evaluation(image_path, software='predict', label_radius=10):
    print('loading full images...')
    image_full_dict = {}
    label_full_dict = {}
    software_full_dict = {}
    for i, im_id in tqdm(enumerate(image_path), total=len(image_path)):
        if software == 'predict' or software == 'fp':
            lbsf_software = './predictions/all_predictions/' + im_id.split('/')[-1][:-3] + 'pred.txt'
        else:
            lbsf_software = im_id[:-3] + software + '.txt'  # 'dozor.txt'
        lbsf = im_id[:-3] + 'manual.txt'
        images_full, labels_full = load_data_full_img(im_id, lbsf, label_radius)
        images_full = (images_full - np.min(images_full)) / (np.max(images_full) - np.min(images_full))
        image_full_dict[im_id] = images_full
        label_full_dict[im_id] = labels_full
        results_software = np.zeros_like(images_full)
        try:
            gts = get_gts(lbsf_software)
            for position in gts:
                if ('albula' in lbsf_software) or ('dozor' in lbsf_software) or ('dials' in lbsf_software):  #
                    results_software[int(float(position[1])), int(float(position[0]))] = 1
                else:
                    results_software[int(float(position[0])), int(float(position[1]))] = 1
        except:
            print(im_id, ' has no predictions')
        image_software = morphology.binary_dilation(results_software, morphology.disk(label_radius)).astype(np.uint8)
        software_full_dict[im_id] = image_software
    w, h = images_full.shape
    print('full image size:', (w, h))
    new_label_full_dict = {}
    new_image_full_dict = {}
    for im_id in image_path:
        new_label_full_dict[im_id] = np.zeros((w, h))
        new_image_full_dict[im_id] = np.zeros((w, h))
        new_label_full_dict[im_id] += label_full_dict[im_id]
        new_image_full_dict[im_id] += software_full_dict[im_id]
    print('evaluation...')
    metrics_test = detection_metrix(np.array(list(new_label_full_dict.values())),
                                    np.array(list(new_image_full_dict.values())))
    prec_avg, rec_avg = metrics_test[0] / metrics_test[2], metrics_test[0] / metrics_test[1]
    f1_avg = 2 * prec_avg * rec_avg / (prec_avg + rec_avg)
    print(f'average precision {prec_avg:.5f}, average recall {rec_avg:.5f}, average f1 score {f1_avg:.5f}')


def mask_possprocess(mask, th=0.4):
    mask_poseprocessed = (mask > th).astype(np.int32)
    for i, mask in enumerate(mask_poseprocessed):
        mask_poseprocessed[i] = morphology.binary_opening(mask, morphology.diamond(3)).astype(np.int32)
    return mask_poseprocessed


original_img_dir = '../../Files_before31Dec2021/'
# original_img_dir = '../../June/'
original_img_dir = '../../Jul21/'
original_img_dir = '../../all_data/'


if __name__ == '__main__':
    splitpath = '../data/all_data/data_split_new.json'
    fullpath = '../data/all_data/data_full_path.json'
    datadir = '../../all_data/'
    save_suffix = ''
    software = 'dials'  # dozor, dials, predict
    with open(splitpath, 'r') as fp:
        vnames = json.load(fp)['test']
    software_evaluation(vnames, software)
