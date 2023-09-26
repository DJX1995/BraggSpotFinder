import numpy as np
import cbf
import cv2
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import glob
import json

# ring_resolution_list = [3.90, 3.67, 3.44, 2.67, 2.25, 2.07, 1.96, 1.93, 1.89, 1.52, 1.47, 1.36, 1.25]
ring_resolution_list = [3.8895, 3.6565, 3.435, 2.665, 2.2465, 2.066, 1.9455, 1.914, 1.880, 1.717, 1.5225, 1.4715, 1.442,
                        1.370, 1.365, 1.2975, 1.2745, 1.260, 1.223, 1.1695, 1.123]

ring_resolution = np.array(ring_resolution_list)


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
        valid = (wavelength <= (2 * ring_resolution))
        new_ring_resolution = ring_resolution[valid]
        ring_radius = detector_distance * np.tan(2 * np.arcsin(wavelength / (2 * new_ring_resolution)))
    pixel_size = content.miniheader['y_pixel_size']
    beam_center = np.array([content.miniheader['beam_center_y'], content.miniheader['beam_center_x']])
    beam_center = beam_center.astype(int)
    # beam_center = image_size // 2
    # beam_center = beam_center.astype(int)
    # beam_center = np.array([1573, 1557])
    if print_all:
        print(f'wavelength: {wavelength}, detector_distance: {detector_distance}, ring_radius: {ring_radius}, \
        pixel_size: {pixel_size} mm, beam_center: {beam_center}')
    return detector_distance, pixel_size, beam_center


def fix_header():
    wavelength = 0.92
    detector_distance = 0.25
    pixel_size = 7.5e-05
    beam_center = np.array([1579, 1611])
    beam_center = beam_center.astype(int)
    return detector_distance, pixel_size, beam_center


def parse_header_old(imf='../Files_before31Dec2021/Mpro_Tel_Mitegen_NoOil6_1231_310.cbf',
                 print_all=False, return_ratio=False):
    content = cbf.read(imf, parse_miniheader=True)
    wavelength = content.miniheader['wavelength']
    if wavelength is None:
        return None
    detector_distance = content.miniheader['detector_distance']
    try:
        ring_radius = detector_distance * np.tan(2 * np.arcsin(wavelength / (2 * ring_resolution)))
    except:
        valid = (wavelength <= (2 * ring_resolution))
        new_ring_resolution = ring_resolution[valid]
        ring_radius = detector_distance * np.tan(2 * np.arcsin(wavelength / (2 * new_ring_resolution)))
    pixel_size = content.miniheader['x_pixel_size']
    ring_radius_pxl = ring_radius / pixel_size
    beam_center = np.array([content.miniheader['beam_center_y'], content.miniheader['beam_center_x']])
    beam_center = beam_center.astype(int)
    image_size = np.array([content.miniheader['pixels_in_y'], content.miniheader['pixels_in_x']])
    # beam_center = image_size // 2
    # beam_center = beam_center.astype(int)
    # beam_center = np.array([1573, 1557])
    if print_all:
        print(f'wavelength: {wavelength}, detector_distance: {detector_distance}, ring_radius: {ring_radius}, \
        pixel_size: {pixel_size} mm, beam_center: {beam_center}')

    if return_ratio:
        return ring_radius, beam_center, image_size, pixel_size
    return ring_radius_pxl, beam_center, image_size


def fix_header_old():
    wavelength = 0.92
    detector_distance = 0.25
    pixel_size = 7.5e-05
    ring_radius = detector_distance * np.tan(2 * np.arcsin(wavelength / (2 * ring_resolution)))
    ring_radius_pxl = ring_radius / pixel_size
    beam_center = np.array([1579, 1611])
    beam_center = beam_center.astype(int)
    image_size = np.array([3269, 3110])
    return ring_radius_pxl, beam_center, image_size


def image_scale(image_gray):
    image_gray[image_gray > 255] = 0.  # artifacts caused by detector?
    image_gray[image_gray < 0] = 0  # grids are -1, artifacts are -2 values?
    image_gray = image_gray / 255
    return image_gray


def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


def generate_empty_image_with_ring(image_id, image_size, width=3):
    # single image
    imf = image_id
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
        if width > 5:
            width = 5
#         print(width)
        mask = abs(distance - r) < width
        image[mask] = 1
    image[image > 1] = 1
    return image * 255


def generate_distorted_ring_image(imf='H2O_EG_125mm_2953_master000001.cbf',
                                  v1=np.array([0.007671, -0.007374, 0.999943]),
                                  v2=np.array([0, 0, 1]),
                                  return_image=False):
    out = parse_header(imf)
    if out is None:
        detector_distance, pixel_size, beam_center = fix_header()
    else:
        detector_distance, pixel_size, beam_center = out
    # get gray scale image and normalize it
    content = cbf.read(imf, parse_miniheader=True)
    image_gray = content.data
    image_gray = image_scale(image_gray) * 255
    # get beam center
    beam_center = beam_center
    # get (x,y,z) of point p0 in distorted detector space
    tmp_y = np.arange(0, image_gray.shape[0]) - beam_center[0]
    tmp_y = np.expand_dims(tmp_y, axis=-1)
    tmp_y = np.repeat(tmp_y, image_gray.shape[1], axis=1)
    tmp_x = np.arange(0, image_gray.shape[1]) - beam_center[1]
    tmp_x = np.expand_dims(tmp_x, axis=0)
    tmp_x = np.repeat(tmp_x, image_gray.shape[0], axis=0)
    tmp_z = np.zeros(tmp_x.shape)
    coords_p0 = np.stack([tmp_x, tmp_y, tmp_z], axis=-1)
    # perform rotation to get (x,y,z) of point p1 in correct detector space
    mat_rotation = rotation_matrix_from_vectors(v1, v2)
    coords_p1 = np.matmul(coords_p0, mat_rotation)
    # get detector distance in pixel value scale
    detector_distance = detector_distance / pixel_size
    # map to 2d (x,y) points p' in correct detector space
    dinominator = detector_distance - coords_p1[:, :, -1]
    coords_p2 = detector_distance * coords_p1 / np.expand_dims(dinominator, axis=-1)
    coords_p2_2d_tmp = coords_p2[:, :, :2]
    coords_p2_2d = coords_p2_2d_tmp[:, :, [1, 0]]
    coords_p2_2d = np.around(coords_p2_2d).astype(int)
    coords_p2_2d[:, :, 0] = coords_p2_2d[:, :, 0] + beam_center[0]
    coords_p2_2d[:, :, 1] = coords_p2_2d[:, :, 1] + beam_center[1]
    # generate ice ring image in correct detector space
    image_ring = generate_empty_image_with_ring(imf, image_gray.shape)
    image_new = np.zeros_like(image_gray)
    # map ring image to distorted detector space
    for i in range(coords_p2_2d.shape[0]):
        for j in range(coords_p2_2d.shape[1]):
            y = coords_p2_2d[i, j][0]
            x = coords_p2_2d[i, j][1]
            if (x < 0) or (y < 0) or (x >= image_ring.shape[1]) or (y >= image_ring.shape[0]):
                continue
            image_new[i, j] = image_ring[y, x]
    if return_image:
        return image_new, image_gray
    return image_new


def traverse_files(path, condition=None):
    if condition != None:
        filepath = path + condition
    else:
        filepath = path + '**/*'
    filenames = []
    for filename in glob.iglob(filepath, recursive=True):
        filenames.append(filename)
    return filenames


if __name__ == "__main__":
    splitpath = '../../data/all_data/data_split.json'
    save_suffix = ''
    with open(splitpath, 'r') as fp:
        filenames = json.load(fp)['test']
    
    savedir = './corrected_new_dynamic_ring_test/'
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    for idx, imf in tqdm(enumerate(filenames)):
        imf = '../' + imf
        img_name = imf.split('/')[-1][:-4]
        # print(img_name)
        image_ring = generate_distorted_ring_image(imf, return_image=False)
        
        cv2.imwrite(f"{savedir}{img_name}_ring.jpg", image_ring)

