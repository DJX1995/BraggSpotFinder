import os
import sys
import numpy as np
import cbf
import re
from skimage import morphology, draw
from skimage.measure import label, regionprops
import logging
import time
from tqdm import tqdm
import json
import h5py
import random
import glob


ring_resolution_list = [3.90, 3.67, 3.44, 2.67, 2.25, 2.07, 1.96, 1.93, 1.89, 1.52, 1.47, 1.36, 1.25]
ring_resolution = np.array(ring_resolution_list)


def parse_header(imf='../Files_before31Dec2021/Mpro_Tel_Mitegen_NoOil6_1231_310.cbf',
                 print_all=False, return_ratio=False):
    content = cbf.read(imf, parse_miniheader=True)
    wavelength = content.miniheader['wavelength']
    if wavelength is None:
        return None
    detector_distance = content.miniheader['detector_distance']
    ring_radius = detector_distance * np.tan(2 * np.arcsin(wavelength / (2 * ring_resolution)))
    pixel_size = content.miniheader['x_pixel_size']
    ring_radius_pxl = ring_radius / pixel_size
    beam_center = np.array([content.miniheader['beam_center_y'], content.miniheader['beam_center_x']])
    beam_center = beam_center.astype(int)
    image_size = np.array([content.miniheader['pixels_in_y'], content.miniheader['pixels_in_x']])
    # beam_center = image_size // 2
    # beam_center = beam_center.astype(int)
    if print_all:
        print(f'wavelength: {wavelength}, detector_distance: {detector_distance}, ring_radius: {ring_radius}, \
        pixel_size: {pixel_size} mm, beam_center: {beam_center}')
    if return_ratio:
        return ring_radius, beam_center, image_size, pixel_size
    return ring_radius_pxl, beam_center, image_size


def fix_header(return_ratio=False):
    wavelength = 0.92
    detector_distance = 0.25
    pixel_size = 7.5e-05
    ring_radius = detector_distance * np.tan(2 * np.arcsin(wavelength / (2 * ring_resolution)))
    ring_radius_pxl = ring_radius / pixel_size
    beam_center = np.array([1579, 1611])
    beam_center = beam_center.astype(int)
    image_size = np.array([3269, 3110])
    if return_ratio:
        return ring_radius_pxl, beam_center, image_size, pixel_size
    return ring_radius_pxl, beam_center, image_size


def get_gts(mannualf='data/Files-31Aug2021/best-but-rings.manual.txt'):
    gts = []
    with open(mannualf, "r") as f:
        f.readline()
        for r in f:
            position = re.split(',| |\t', r)
            if position:
                gts += [[int(float(position[0])), int(float(position[1]))]]
    return np.asarray(gts)


def image_scale(image_gray):
    image_gray[image_gray > 255] = 0  # artifacts caused by detector?
    image_gray[image_gray < 0] = 0  # grids are -1, artifacts are -2 values?
    image_gray = image_gray / 255
    return image_gray


def load_data_full_img(imf='../data/Files-21Oct2021/mx308820-2_hetB_B-2_026.cbf',
                       lbsf='../data/Files-21Oct2021/mx308820-2_hetB_B-2_026_manual.txt',
                       label_radius=10,
                       return_gt=False):
    # lbsf = lbsf.replace('manual',label_suffix)
    image_gray = cbf.read(imf).data
    image_gray = image_scale(image_gray)
    labels = np.zeros_like(image_gray)
    gts = get_gts(lbsf)
    for position in gts:
        if ('albula' in lbsf) or ('dozor' in lbsf) or ('dials' in lbsf):  #
            labels[int(float(position[1])), int(float(position[0]))] = 1
        else:
            labels[int(float(position[0])), int(float(position[1]))] = 1
    labels = morphology.binary_dilation(labels, morphology.diamond(label_radius)).astype(np.uint8)
    w, h = image_gray.shape
    # print('image size: ', w, h)
    if return_gt:
        return image_gray, labels, gts
    return image_gray, labels


def load_data_full_img_disk(imf='../data/Files-21Oct2021/mx308820-2_hetB_B-2_026.cbf',
              lbsf='../data/Files-21Oct2021/mx308820-2_hetB_B-2_026_manual.txt', label_radius=5):
    # lbsf = lbsf.replace('manual',label_suffix)
    image_gray = cbf.read(imf).data
    if image_gray.max() > 60000:
        image_gray[image_gray == image_gray.max()] = 0
    image_gray[image_gray > 255] = 255
    image_gray = image_gray / 255
    labels = np.zeros_like(image_gray)
    gts = get_gts(lbsf)
    for position in gts:
        if ('albula' in lbsf) or ('dozor' in lbsf) or ('dials' in lbsf):  #
            labels[int(float(position[1])), int(float(position[0]))] = 1
        else:
            labels[int(float(position[0])), int(float(position[1]))] = 1
    labels = morphology.binary_dilation(labels, morphology.disk(5)).astype(np.uint8)
    w, h = image_gray.shape
    # print('image size: ', w, h)
    return image_gray, labels


def load_data(imf='../data/Files-21Oct2021/mx308820-2_hetB_B-2_026.cbf',
              lbsf='../data/Files-21Oct2021/mx308820-2_hetB_B-2_026_manual.txt',
              label_radius=10,
              skip_negative=True,
              radius_pos=False,
              mask_ring=False):
    # lbsf = lbsf.replace('manual',label_suffix)
    image_gray = cbf.read(imf).data
    image_gray = image_scale(image_gray)

    out = parse_header(imf, return_ratio=True)
    if out is None:
        radius, center, _, pixel_size = fix_header(return_ratio=True)
    else:
        radius, center, _, pixel_size = out
    w, h = image_gray.shape
    assert len(ring_resolution) == len(radius)
    pos_r = np.zeros(shape=(w, h, len(radius)))
    i_coords, j_coords = np.meshgrid(range(w), range(h), indexing='ij')
    gids = np.stack([i_coords, j_coords], axis=-1)
    distance = np.linalg.norm(gids - center, axis=-1)
    distance = distance * pixel_size
    for i in range(len(radius)):
        pos_r[:, :, i] = abs(distance - radius[i])  # / radius[i]
    # max_r = 100 * pixel_size
    # pos_r = np.clip(pos_r, 0., max_r)

    im = []
    lbs = []
    labels = np.zeros_like(image_gray)
    gts = get_gts(lbsf)
    for position in gts:
        if ('albula' in lbsf) or ('dozor' in lbsf) or ('dials' in lbsf):  #
            labels[int(float(position[1])), int(float(position[0]))] = 1
        else:
            labels[int(float(position[0])), int(float(position[1]))] = 1
    # labels = morphology.binary_dilation(labels, morphology.diamond(label_radius)).astype(np.uint8)
    w, h = image_gray.shape
    # print('image size: ', w, h, end=' ')

    if mask_ring:
        i_coords, j_coords = np.meshgrid(range(w), range(h), indexing='ij')
        gids = np.stack([i_coords, j_coords], axis=-1)
        distance = np.linalg.norm(gids - center, axis=-1)
        # radius, center are known
        image_mask = np.ones((w, h))
        for r in radius:
            mask = abs(distance - r) < 5
            image_mask[mask] = 0
        image_gray = image_gray * image_mask
    xy_position = []
    r_position = []
    for xi, x in enumerate(range(w % 512 // 2, w // 512 * 512 + w % 512 // 2, 512)):
        for yi, y in enumerate(range(h % 512 // 2, h // 512 * 512 + h % 512 // 2, 512)):
            lbs_patch = labels[x:x + 512, y:y + 512]
            if skip_negative and lbs_patch.max() < 1:
                continue
            lbs_patch = morphology.binary_dilation(lbs_patch, morphology.diamond(label_radius)).astype(np.uint8)
            lbs.append(lbs_patch)
            image_patch = image_gray[x:x + 512, y:y + 512]
            pos_r_patch = pos_r[x:x + 512, y:y + 512, :]
            im.append(image_patch)
            r_position.append(pos_r_patch)
            xy_position.append([xi, yi])
    im = np.asarray(im).reshape([-1, 512, 512, 1])
    lbs = np.asarray(lbs).reshape([-1, 512, 512, 1])
    xy_position = np.asarray(xy_position).reshape([-1, 2])
    r_position = np.asarray(r_position).reshape([-1, 512, 512, len(radius)])
    # r_position = r_position * pixel_size
    xy_position = np.asarray(xy_position).reshape([-1, 2])
    if radius_pos:
        return im, lbs, gts, xy_position, r_position, image_gray, labels
    return im, lbs, gts, xy_position, image_gray, labels


def prepare_data(datadir, savedir, skip_negative=False, save_suffix='', split=None): 
    im_all = []
    lbs_all = []
    im_id_all = []
    im_path_all = []
    im_pos_all = []
    im_pos_r_all = []
    all_files = traverse_files(datadir, condition=None)
    image_files = [f for f in all_files if f.endswith('cbf')]
    count = 0
    for i, imf in tqdm(enumerate(image_files),
                       total=len(image_files),
                       desc='convert cbf image files to numpy arrays'):
        lbsf = imf[:-4] + '.manual.txt'
        if lbsf in all_files:
            if imf in split['test']:
                im, lb, gts, xy_position, r_position, image_gray, labels = load_data(imf=imf, lbsf=lbsf, label_radius=7, skip_negative=skip_negative, radius_pos=True, mask_ring=False)
            else:
                im, lb, gts, xy_position, r_position, image_gray, labels = load_data(imf=imf, lbsf=lbsf, label_radius=7, skip_negative=skip_negative, radius_pos=True, mask_ring=False)
#             im, lb, gts, xy_position, r_position, image_gray, labels = load_data_mask_ring(imf=imf, lbsf=lbsf, label_radius=10, skip_negative=skip_negative, radius_pos=True, mask_ring=False)
            im_all.append(im)
            lbs_all.append(lb)
            im_pos_all.append(xy_position)
#             im_pos_r_all.append(r_position)
            im_id_all.append([imf.split('/')[-1]] * len(im))
#             im_path_all.append([imf] * len(im))
            count += len(im)
    im_all = np.concatenate(im_all)
    lbs_all = np.concatenate(lbs_all)
    im_pos_all = np.concatenate(im_pos_all)
#     im_pos_r_all = np.concatenate(im_pos_r_all)
    im_id_all = np.concatenate(im_id_all)
    print(f'data size: {len(im_all)}')
    # for index in range(len(set(list(im_id_cb)))):
    for i, im_id in tqdm(enumerate(set(list(im_id_all)))):
        # print(i, im_id)
        with h5py.File(savedir + f'data_combine{save_suffix}.h5py', 'a') as hf:
            grp = hf.create_group(im_id)
            idx = []
            for i, name in enumerate(im_id_all):
                if name == im_id:
                    idx.append(i)
            grp['im'] = im_all[idx]
            grp['lbs'] = lbs_all[idx]
            grp['im_pos'] = im_pos_all[idx]
#             grp['im_pos_r'] = im_pos_r_all[idx]


def train_test_split(filenames, split=5, random_seed=0):
    random.seed(random_seed)
    sample_num = len(filenames)
    split = sample_num // split
    indices = list(range(sample_num))
    random.shuffle(indices)
    train_idx = indices[split:]
    test_idx = indices[:split]
    sample_train = [filenames[idx] for idx in train_idx]
    sample_test = [filenames[idx] for idx in test_idx]
    return sample_train, sample_test
    
    
def traverse_files(path, condition=None):
    if condition != None:
        filepath = path + condition
    else:
        filepath = path + '**/*'
    filenames = []
    for filename in glob.iglob(filepath, recursive=True):
        filenames.append(filename)
    return filenames


def load_data_lbs(imf='../data/Files-21Oct2021/mx308820-2_hetB_B-2_026.cbf',
              lbsf='../data/Files-21Oct2021/mx308820-2_hetB_B-2_026_manual.txt',
              skip_negative=True):
    image_gray = cbf.read(imf).data
    lbs = []
    labels = np.zeros_like(image_gray)
    gts = get_gts(lbsf)
    for position in gts:
        if ('albula' in lbsf) or ('dozor' in lbsf) or ('dials' in lbsf):  #
            labels[int(float(position[1])), int(float(position[0]))] = 1
        else:
            labels[int(float(position[0])), int(float(position[1]))] = 1
    w, h = image_gray.shape
    for xi, x in enumerate(range(w % 512 // 2, w // 512 * 512 + w % 512 // 2, 512)):
        for yi, y in enumerate(range(h % 512 // 2, h // 512 * 512 + h % 512 // 2, 512)):
            lbs_patch = labels[x:x + 512, y:y + 512]
            if skip_negative and lbs_patch.max() < 1:
                continue
            lbs.append(lbs_patch)
    lbs = np.asarray(lbs).reshape([-1, 512, 512, 1])
    return lbs, gts


def dataset_statistics(splitpath, fullpath, split='train', skip_negative=True):
    with open(fullpath, 'r') as fp:
        full_path_dict = json.load(fp)
    with open(splitpath, 'r') as fp:
        vnames = json.load(fp)
    count_images = 0
    count_patches = 0
    count_spots = 0
    for i, vname in tqdm(enumerate(vnames[split]),
                       total=len(vnames[split]),
                       desc='convert cbf image files to numpy arrays'):
        im_name = vname.split('/')[-1]
        imf = full_path_dict[im_name]
        lbsf = imf[:-4] + '.manual.txt'
        lbs, gts = load_data_lbs(imf=imf, lbsf=lbsf, skip_negative=skip_negative,)
        count_images += 1
        count_patches += len(lbs)
        count_spots += len(gts)
    print(f'{split} set has {count_images} images, {count_patches} patches and {count_spots} spots')    
    
    
    
if __name__ == '__main__':
    datadir = '../../all_data/'
    condition = '**/*.cbf'
    savedir = '../data/all_data/'
    filenames = traverse_files(datadir, condition)
    # image name --- image path mapping
    full_path_dict = {}
    for fname in filenames:
        full_path_dict[fname.split('/')[-1]] = fname
    with open(savedir + 'data_full_path.json', 'w') as fp:
        json.dump(full_path_dict, fp)
    # train, test split based on image names
    sample_train, sample_test = train_test_split(filenames, split=5, random_seed=0)
    split = {'train': sample_train, 'test': sample_test}
    with open(savedir + 'data_split.json', 'w') as fp:
        json.dump(split, fp)
    # generate rgb train, test images based on the previous splits
    with open('../data/all_data/data_split.json', 'r') as fp:
        split = json.load(fp)
    prepare_data(datadir, savedir=savedir, skip_negative=True, save_suffix='_without_negative_r7', split=split)

#     with open(savedir + 'data_split.json', 'r') as fp:
#         split = json.load(fp)
    


