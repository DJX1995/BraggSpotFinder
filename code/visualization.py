import numpy as np
import nibabel as nib
import cbf
import cv2
import os
from scipy import ndimage as ndi
from tqdm import tqdm
import json

from skimage import morphology, draw
from skimage.measure import label, regionprops
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage import measure
from util import draw_diamond, get_prop_bbox, non_max_suppression, get_metrics, get_box_center, detection_metrix
from data_preprocess import load_data_full_img, get_gts
from ice_ring_correction.ring_correction import parse_header_old, fix_header_old


ring_resolution_list = [3.90, 3.67, 3.44, 2.67, 2.25, 2.07, 1.96, 1.93, 1.89, 1.52, 1.47, 1.36, 1.25]

ring_img_dir = './ice_ring_correction/corrected_new_dynamic_ring_test2/'


def parse_header(imf='../Files_before31Dec2021/Mpro_Tel_Mitegen_NoOil6_1231_310.cbf',
                 print_all=False):
    content = cbf.read(imf, parse_miniheader=True)
    wavelength = content.miniheader['wavelength']
    if wavelength is None:
        return None
    detector_distance = content.miniheader['detector_distance']
    try:
        ring_radius = detector_distance * np.tan(2 * np.arcsin(wavelength / (2 * ring_resolution)))
    except:
        valid = wavelength <= (2 * ring_resolution)
        new_ring_resolution = ring_resolution[valid]
        ring_radius = detector_distance * np.tan(2 * np.arcsin(wavelength / (2 * new_ring_resolution)))
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
    return ring_radius_pxl, beam_center, image_size


def fix_header():
    wavelength = 0.92
    detector_distance = 0.25
    pixel_size = 7.5e-05
    ring_radius = detector_distance * np.tan(2 * np.arcsin(wavelength / (2 * ring_resolution)))
    ring_radius_pxl = ring_radius / pixel_size
    beam_center = np.array([1579, 1611])
    beam_center = beam_center.astype(int)
    image_size = np.array([3269, 3110])
    return ring_radius_pxl, beam_center, image_size


def draw_image_ring(image_id, datadir, image_before):
    imf = image_id
    image_size = image_before.shape[:2]
    out = parse_header_old(imf)
    if out is None:
        radius, center, _, ring_radius_width = fix_header_old()
    else:
        radius, center, _, ring_radius_width = out
    image = np.zeros(image_size)
    i_coords, j_coords = np.meshgrid(range(image_size[0]), range(image_size[1]), indexing='ij')
    gids = np.stack([i_coords, j_coords], axis=-1)
    distance = np.linalg.norm(gids - center, axis=-1)
    # radius, center are known
    for i, r in enumerate(radius):
        width = (ring_radius_width[i] / 2)
        if width < 4:
            width = 4
        mask = abs(distance - r) < width
        image[mask] = 1
    image_tmp = image
    image_before[:, :, 0] += image_tmp
    image_before[:, :, 1] += image_tmp
    image_before[:, :, 2] += image_tmp
    image_before[image_before > 1] = 1
    return image_before * 255


def draw_image_ring_1d(image_id, datadir, image_before):
    imf = image_id
    image_size = image_before.shape
    out = parse_header_old(imf)
    if out is None:
        radius, center, _, ring_radius_width = fix_header_old()
    else:
        radius, center, _, ring_radius_width = out
    image = np.zeros(image_size)
    i_coords, j_coords = np.meshgrid(range(image_size[0]), range(image_size[1]), indexing='ij')
    gids = np.stack([i_coords, j_coords], axis=-1)
    distance = np.linalg.norm(gids - center, axis=-1)
    # radius, center are known
    for i, r in enumerate(radius):
        width = (ring_radius_width[i] / 2)
        if width < 4:
            width = 4
        mask = abs(distance - r) < 4 # width
        image[mask] = 1
    image_tmp = image
    image_before += image_tmp
    image_before[image_before > 1] = 1
    return image_before


def mask_possprocess(mask, th=0.4):
    mask_poseprocessed = (mask > th).astype(np.int32)
    # for i, mask in enumerate(mask_poseprocessed):
    #     mask_poseprocessed[i] = morphology.binary_opening(mask, morphology.diamond(3)).astype(np.int32)
    return mask_poseprocessed


def draw_with_annotation(datadir, save_img_dir, save_suffix, image_id, software, label_radius=10):
    '''
    :param datadir: dir has image files
    :param save_img_dir: dir to save outputs
    :param image_id: image names
    :param software: dozor, dials, or predict
    :param label_radius:
    '''
    print('loading full images ......')
    image_id = image_id[:10]
    image_full_dict = {}
    label_full_dict = {}
    software_full_dict = {}
    for im_id in image_id:      
        if software == 'predict' or software == 'fp':
            lbsf_software = './predictions/fps_/' + im_id.split('/')[-1][:-3] + 'fp.txt'
        else:
            lbsf_software = im_id[:-3] + software + '.txt'  # 'dozor.txt'
        lbsf = im_id[:-4] + '.manual.txt'
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
            pass
        image_software = morphology.binary_dilation(results_software, morphology.diamond(label_radius)).astype(np.uint8)
        software_full_dict[im_id] = image_software
    w, h = images_full.shape
    print('generating TPs, FPs, FN ......')
    tmp = np.array(list(software_full_dict.values()))
    result_TP = np.zeros_like(tmp)
    result_FP = np.zeros_like(tmp)
    result_FN = np.zeros_like(tmp)
    result_GT = np.zeros_like(tmp)

    for i, im_id in enumerate(image_id):
        res = software_full_dict[im_id]
        lbs = label_full_dict[im_id]
        labeled_mask = label(res)
        props = regionprops(labeled_mask)
        for x in props:
            if lbs[:, :][int(x.centroid[0]), int(x.centroid[1])] < 1:
                result_FP[i][labeled_mask == x.label] = 1
            else:
                result_TP[i][labeled_mask == x.label] = 1
        labeled_lbs = label(lbs[:, :])  # label different regions
        props = regionprops(labeled_lbs)
        for x in props:
            if res[int(float(x.centroid[0])), int(float(x.centroid[1]))] <= 0:
                result_FN[i][labeled_lbs == x.label] = 1
    result_TP = result_TP - morphology.binary_erosion(result_TP, np.expand_dims(morphology.diamond(2), 0)).astype(
        np.uint8)
    result_FP = result_FP - morphology.binary_erosion(result_FP, np.expand_dims(morphology.diamond(2), 0)).astype(
        np.uint8)
    result_FN = result_FN - morphology.binary_erosion(result_FN, np.expand_dims(morphology.diamond(2), 0)).astype(
        np.uint8)
    result_TP[result_TP > 0] = 1
    result_FP[result_FP > 0] = 1
    result_FN[result_FN > 0] = 1
    # result_GT = morphology.binary_erosion(result_GT, np.expand_dims(morphology.diamond(5), 0)).astype(np.uint8)

    print('merging full images with TP, FP, FN annotations ......')
    new_image_full_dict = {}
    for idx, im_id in enumerate(image_id):
        full_image = np.zeros((w, h, 3))
        full_image_norm = cv2.normalize(image_full_dict[im_id], dst=None, alpha=0, beta=4, norm_type=cv2.NORM_MINMAX)
        full_image[:, :, 0] += full_image_norm
        full_image[:, :, 1] += full_image_norm
        full_image[:, :, 2] += full_image_norm
        full_image[:, :, 2] += result_TP[idx]
#         full_image[:, :, 1] += result_FN[idx]
        full_image[:, :, 2] += result_FP[idx]
        full_image[:, :, 1] += result_FP[idx]
        # full_image[:, :, 2] += result_FP[idx]
        # full_image[:, :, 2] += result_GT[idx]
        # full_image[:, :, 1] += result_GT[idx]
        # full_image[:, :, 0] += result_GT[idx]
        new_image_full_dict[im_id] = full_image

    if save_img_dir != None:
        print('saving images ......')
        if not os.path.exists(save_img_dir):
            os.makedirs(save_img_dir)
        for im_id in image_id:
            result_image = new_image_full_dict[im_id]
            result_image[result_image > 1] = 1
            save_id = im_id.split('/')[-1]
            cv2.imwrite(f'{save_img_dir}{save_id[:-4]}_{save_suffix}_{software}.jpg', result_image * 255)

                
def draw_with_prediction(dirname, datadir, save_img_dir, save_suffix, image_id, draw_ring=False, mask_ring=False):
    '''
    :param dirname: dir has prediction files
    :param datadir: dir has image data files
    :param save_img_dir: dir to save outputs
    :param image_id: image names
    '''
    patch_id, patch_pos = np.load(dirname + 'test_ids.npy'), np.load(dirname + 'test_pos_ids.npy')
    labels = nib.load(dirname + 'testlables_save.nii').get_fdata()
    result = nib.load(dirname + 'vsoftmax_processed.nii').get_fdata()
    result = result[:len(labels)]
    pred = nib.load(dirname + 'testsoftnax_save.nii').get_fdata()
    result = mask_possprocess(pred, th=0.4)
    im_topk = 61
    curr_id = patch_id[0]
    count = 0
    for idx in range(1, len(patch_id)):
        if curr_id != patch_id[idx]:
            count += 1
            curr_id = patch_id[idx]
            if count == im_topk:
                break
    patch_topk = idx
    patch_id = patch_id[:idx]
    patch_pos = patch_pos[:idx]
    labels = labels[:idx]
    result = result[:idx]
    image_id = np.unique(patch_id)
    
    choose_id = 'JLJ731_21_7875_18'
    image_id_new = []
    for im_id in image_id:
        if (choose_id in im_id):
            image_id_new.append(im_id)
    image_id = image_id_new

    image_full_dict = {}
    label_full_dict = {}
    ring_full_dict = {}
    for im_id in image_id:
        imf = full_path_dict[im_id]
        lbsf = imf[:-3] + 'manual.txt'
        # lbsf = im_id[:-4] + '.manual.txt'
        # print(imf, lbsf)
        images_full, labels_full = load_data_full_img(imf, lbsf)
        # images_full, labels_full = load_data_full_img_disk(imf, lbsf)
        images_full = (images_full - np.min(images_full)) / (np.max(images_full) - np.min(images_full))
        image_full_dict[im_id] = images_full
        label_full_dict[im_id] = labels_full
        out = parse_header(imf)
        if out is None:
            radius, center, _ = fix_header()
        else:
            radius, center, _ = out
        ring_full_dict[im_id] = [radius, center]
    img_size = images_full.shape
    w, h = images_full.shape
    print('full image size:', (w, h))
    # image_id = np.unique(patch_id)
    new_label_full_dict = {}
    new_image_full_dict = {}
    for im_id in image_id:
        new_label_full_dict[im_id] = np.zeros((w, h, 1))
        new_image_full_dict[im_id] = np.zeros((w, h))

    for idx in range(len(patch_pos)):
        im_id = patch_id[idx]
        if not (choose_id in im_id):
            continue
        pos_x, pos_y = patch_pos[idx]
        patchx1, patchx2 = w % 512 // 2 + pos_x * 512, w % 512 // 2 + (pos_x + 1) * 512
        patchy1, patchy2 = h % 512 // 2 + pos_y * 512, h % 512 // 2 + (pos_y + 1) * 512
        patchx1 = int(patchx1)
        patchx2 = int(patchx2)
        patchy1 = int(patchy1)
        patchy2 = int(patchy2)
        new_label_full_dict[im_id][patchx1:patchx2, patchy1:patchy2] += labels[idx]
        new_image_full_dict[im_id][patchx1:patchx2, patchy1:patchy2] += result[idx]
    print('generating TPs, FPs, FN ......')
    tmp = np.array(list(new_image_full_dict.values()))
    result_TP = np.zeros_like(tmp)
    result_FP = np.zeros_like(tmp)
    result_FN = np.zeros_like(tmp)
    result_GT = np.zeros_like(tmp)

    for i, im_id in enumerate(image_id):
        img = image_full_dict[im_id]
        res = new_image_full_dict[im_id]
        labels = label(res)  # label different regions
        props = regionprops(labels)
        # split large region
        for x in props:
            if (x.area > 450) and ((x.bbox[2] - x.bbox[0]) > 40 or (x.bbox[3] - x.bbox[1]) > 40):
                res_tmp = res[x.bbox[0]:x.bbox[2], x.bbox[1]:x.bbox[3]]
                distance = ndi.distance_transform_edt(res_tmp)
                # coords = peak_local_max(distance, footprint=np.ones((10, 10)), labels=res)
                coords = peak_local_max(distance, min_distance=7, labels=res_tmp)
                mask = np.zeros(distance.shape, dtype=bool)
                mask[tuple(coords.T)] = True
                markers, _ = ndi.label(mask)
                labels_tmp = watershed(-distance, markers, mask=res_tmp)
                res[x.bbox[0]:x.bbox[2], x.bbox[1]:x.bbox[3]] = labels_tmp
        # remove small region
        labels = label(res)  # label different regions
        props = regionprops(labels)
        for x in props:
            if (x.area < 60) or ((x.bbox[2] - x.bbox[0]) < 9) or ((x.bbox[3] - x.bbox[1]) < 9):
                res[tuple(x.coords.T)] = 0.
                    
        if mask_ring:
            ring_img_full_path = ring_img_dir + im_id[:-4] + '_ring.jpg'
            ring_img = cv2.imread(ring_img_full_path, cv2.IMREAD_GRAYSCALE)
            if ring_img is None:
                print(ring_img_full_path)
                continue
            ring_img = ring_img / 255
            ring_img[ring_img > 0.5] = 1  # artifacts caused by detector?
            ring_img[ring_img < 0.5] = 0  # grids are -1, artifacts are -2 values?
            labels = label(res)  # label different regions
            props = regionprops(labels)
            for x in props:
                distance_to_center = x.centroid - center
                distance_to_center = np.sqrt(np.sum(distance_to_center ** 2))
                if (distance_to_center - 60.) < 0:
                    res[tuple(x.coords.T)] = 0.
                try:
                    if ring_img[int(x.centroid[0]), int(x.centroid[1])] > 0:
                        res[tuple(x.coords.T)] = 0.
                except:
                    print(x.centroid, ring_img.shape)
                    continue
        lbs = new_label_full_dict[im_id]
        labeled_mask = label(res)
        props = regionprops(labeled_mask)
        for x in props:
            if lbs[:, :, 0][int(x.centroid[0]), int(x.centroid[1])] < 1:
                result_FP[i][labeled_mask == x.label] = 1
            else:
                result_TP[i][labeled_mask == x.label] = 1
        labeled_lbs = label(lbs[:, :, 0])  # label different regions
        props = regionprops(labeled_lbs)
        # new visual
        for x in props:
            if res[int(float(x.centroid[0])), int(float(x.centroid[1]))] <= 0:
                result_FN[i][labeled_lbs == x.label] = 1
    result_TP = result_TP - morphology.binary_erosion(result_TP, np.expand_dims(morphology.diamond(3), 0)).astype(
        np.uint8)
    result_FP = result_FP - morphology.binary_erosion(result_FP, np.expand_dims(morphology.diamond(3), 0)).astype(
        np.uint8)
    result_FN = result_FN - morphology.binary_erosion(result_FN, np.expand_dims(morphology.diamond(3), 0)).astype(
        np.uint8)
    result_TP[result_TP > 0] = 1
    result_FP[result_FP > 0] = 1
    result_FN[result_FN > 0] = 1
    # result_GT = morphology.binary_erosion(result_GT, np.expand_dims(morphology.diamond(5), 0)).astype(np.uint8)
    # ring covered by the TP,FP,FN
    print('merging full images with TP, FP, FN annotations ......')
    new_image_full_dict = {}
    for idx, im_id in enumerate(image_id):
        full_image = np.zeros((w, h, 3))
        imf = full_path_dict[im_id]
        image_full_org = image_full_dict[im_id]
        image_full_ring = draw_image_ring_1d(imf, datadir, image_full_org)
        full_image_norm = cv2.normalize(image_full_ring, dst=None, alpha=0, beta=4, norm_type=cv2.NORM_MINMAX)
        full_image_norm[full_image_norm > 1] = 1
        full_image[:, :, 0] += full_image_norm
        full_image[:, :, 1] += full_image_norm
        full_image[:, :, 2] += full_image_norm
        # FP pred
        full_image[:, :, 2][result_FP[idx]>=1] = result_FP[idx][result_FP[idx]>=1]
        full_image[:, :, 1][result_FP[idx]>=1] = result_FP[idx][result_FP[idx]>=1]
        full_image[:, :, 0][result_FP[idx]>=1] = 0
        # TP
        full_image[:, :, 2][result_TP[idx]>=1] = result_TP[idx][result_TP[idx]>=1]
        full_image[:, :, 1][result_TP[idx]>=1] = 0
        full_image[:, :, 0][result_TP[idx]>=1] = 0
        # FN
        full_image[:, :, 1][result_FN[idx]>=1] = result_FN[idx][result_FN[idx]>=1]
        full_image[:, :, 2][result_FN[idx]>=1] = 0
        full_image[:, :, 0][result_FN[idx]>=1] = 0
        
        new_image_full_dict[im_id] = full_image
        # ring above the TP,FP,FN
    if save_img_dir != None:
        print('saving images ......')
        if mask_ring:
            save_img_dir += 'pred_with_mask/'
        else:
            save_img_dir += 'pred_no_mask/'
        if not os.path.exists(save_img_dir):
            os.makedirs(save_img_dir)
        for im_id in image_id:
            imf = full_path_dict[im_id]
            result_image = new_image_full_dict[im_id]
            result_image[result_image > 1] = 1
            # save_id = im_id.split('/')[-1]
            cv2.imwrite(f'{save_img_dir}{im_id[:-4]}_ring_{save_suffix}_draw_ring_mask_ring.jpg', result_image * 255)


def draw_raw_image_with_ring(datadir, save_img_dir, save_suffix, image_id,
                             with_gt=False, draw_ring=False, label_radius=10):
    '''
    :param datadir: dir has image files
    :param save_img_dir: dir to save outputs
    :param image_id: image names
    :param with_gt: whether to plot the GT position
    '''
    print('loading full images ......')
    image_full_dict = {}
    label_full_dict = {}
    for im_id in image_id:
        lbsf = im_id[:-4] + '.manual.txt'
        images_full, labels_full = load_data_full_img(im_id, lbsf, label_radius)
        images_full = (images_full - np.min(images_full)) / (np.max(images_full) - np.min(images_full))
        image_full_dict[im_id] = images_full
        label_full_dict[im_id] = labels_full
    w, h = images_full.shape
    if with_gt:
        print('merging full images with GT annotations ......')
        result_GT = np.array(list(label_full_dict.values()))
        result_GT = result_GT - morphology.binary_erosion(result_GT, np.expand_dims(morphology.diamond(3), 0)).astype(
            np.uint8)
        save_suffix += 'with_gt'
    new_image_full_dict = {}
    for idx, im_id in enumerate(image_id):
        full_image = np.zeros((w, h, 3))
        full_image_norm = cv2.normalize(image_full_dict[im_id], dst=None, alpha=0, beta=4,
                                        norm_type=cv2.NORM_MINMAX)
        full_image[:, :, 0] += full_image_norm
        full_image[:, :, 1] += full_image_norm
        full_image[:, :, 2] += full_image_norm
        if with_gt:
            full_image[:, :, 2] += result_GT[idx]
            full_image[:, :, 1] += result_GT[idx]
            # full_image[:, :, 0] += result_GT[idx]
        new_image_full_dict[im_id] = full_image

    if save_img_dir != None:
        print('saving images ......')
        if not os.path.exists(save_img_dir):
            os.makedirs(save_img_dir)
        for im_id in image_id:
            result_image = new_image_full_dict[im_id]
            result_image[result_image > 1] = 1

            im_id = im_id.split('/')[-1]
            cv2.imwrite(f'{save_img_dir}{im_id[:-4]}_raw_{save_suffix}.jpg', result_image * 255)
            if draw_ring:
                image_ring = draw_image_ring(im_id, datadir, result_image)
                if image_ring is not None:
                    cv2.imwrite(f'{save_img_dir}{im_id[:-4]}_raw_ring_{save_suffix}.jpg', image_ring)

                    
def traverse_files(path, condition=None):
    if condition != None:
        filepath = path + condition
    else:
        filepath = path + '**/*'
    filenames = []
    for filename in glob.iglob(filepath, recursive=True):
        filenames.append(filename)
    return filenames

            
if __name__ == '__main__':
    import glob
    import os
    
    global full_path_dict
    dirname = '../experiment/bnl_unet_all_data_0.88143/'
    datadir = '../../all_data/'
#     save_img_dir = './predictions/fp_with_post_processing/'
    save_img_dir = '../visualization/image_software_all_test/corrected_dynamic_rings_old3/'
    save_suffix = 'no_correction'
    software = 'predict'  # dozor, dials, predict

    filenames = traverse_files(datadir)
    with open('../data/all_data/data_full_path.json', 'r') as fp:
        full_path_dict = json.load(fp)
    splitpath = '../data/all_data/data_split.json'
    datadir = '../../all_data/'
    with open(splitpath, 'r') as fp:
        vnames = json.load(fp)
    image_id = vnames['test']
    #     image_id = [sample.split('/')[-1] for sample in vnames['test']]
    #     print(vnames['test'])
#     draw_raw_image_with_ring(datadir, save_img_dir, save_suffix, vnames['test'], with_gt=True, draw_ring=False)
    
#     save_prediction(dirname, datadir, save_img_dir, save_suffix, image_id=image_id, draw_ring=False,
#                          mask_ring=True)
#     draw_with_annotation(datadir, save_img_dir, save_suffix, image_id=image_id, software=software)
    draw_with_prediction(dirname, datadir, save_img_dir, save_suffix, image_id=image_id, draw_ring=True,
                             mask_ring=False)
#     draw_with_prediction3(dirname, datadir, save_img_dir, save_suffix, draw_ring=False, post_process=False)
    
#     save_img_dir = '../visualization/image_software_all_test/pred_wo_post_process2/tmp/'
#     draw_with_prediction3(dirname, datadir, save_img_dir, save_suffix, draw_ring=False, mask_ring=True)
    
