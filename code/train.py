import os
import sys
import numpy as np
import random
import copy
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from util import *
from tqdm import tqdm
from dataset import dataset, dataset_h5py, collate_fn
from config import config
from unet import seg_model
from evaluation import model_evaluation
import json


# Set device
gpu_index = 0
device = torch.device(f"cuda:{gpu_index}" if torch.cuda.is_available() else "cpu")
# device = 'cpu'

seed = 10
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


def train_epoch(epoch, main_model, train_loader, optimizer):
    '''
    :param epoch:
    :param main_model:
    :param train_loader:
    :param optimizer:
    :return:
    '''
    main_model.train()
    optimizer.zero_grad()
    epoch_loss = []
    ce_loss = []
    dc_loss = []
    rgl_loss = []
    start = time.time()
    for batch_idx, batch_data in tqdm(enumerate(train_loader), desc='training'):
        image_batch, label_batch, image_pos, image_id = batch_data
        image_batch, label_batch = image_batch.to(device), label_batch.to(device)
        out_feature, loss = main_model(image_batch, label_batch)
        total_loss = 100 * loss['ce_loss'] + 0. * loss['dc_loss'] + 0. * loss['rgl_loss']
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        epoch_loss.append(total_loss.item())
        ce_loss.append(loss['ce_loss'].item())
        dc_loss.append(loss['dc_loss'].item())
        rgl_loss.append(loss['rgl_loss'].item())
    end = time.time()
    time_cost = end - start
    epoch_loss = sum(epoch_loss) / len(epoch_loss)
    ce_loss = sum(ce_loss) / len(ce_loss)
    dc_loss = sum(dc_loss) / len(dc_loss)
    rgl_loss = sum(rgl_loss) / len(rgl_loss)
    logger.info(
        f'Epoch {epoch}, Loss {epoch_loss:.10f}, ce_loss {ce_loss:.5f}, '
        f'dc_loss {dc_loss:.5f}, '
        f'rgl_loss {rgl_loss:.5f}, time cost {time_cost / 60:.2f} minutes')
    return epoch_loss


def evaluate_epoch(epoch, main_model, data_loader, split, full_path_dict, return_data=False):
    main_model.eval()

    with torch.no_grad():
        epoch_loss = []
        ce_loss = []
        dc_loss = []
        rgl_loss = []
        mask_pred = []
        mask_gt = []
        image_batchs = []
        label_batchs = []
        image_pos_batchs = []
        image_id_batchs = []
        start = time.time()
        for batch_idx, batch_data in tqdm(enumerate(data_loader), desc=split):
            image_batch, label_batch, image_pos, image_id = batch_data
            image_batch, label_batch = image_batch.to(device), label_batch.to(device)
            out_feature, loss = main_model(image_batch, label_batch)
            image_batchs.append(image_batch)
            label_batchs.append(label_batch)
            image_pos_batchs.append(image_pos)
            image_id_batchs.extend(image_id)
            total_loss = loss['ce_loss'] + 0. * loss['dc_loss'] + 0. * loss['rgl_loss']
            epoch_loss.append(total_loss.item())
            ce_loss.append(loss['ce_loss'].item())
            dc_loss.append(loss['dc_loss'].item())
            rgl_loss.append(loss['rgl_loss'].item())
            mask_pred.append(out_feature.sigmoid())
            mask_gt.append(label_batch)
        end = time.time()
        time_cost = end - start
        epoch_loss = sum(epoch_loss) / len(epoch_loss)
        ce_loss = sum(ce_loss) / len(ce_loss)
        dc_loss = sum(dc_loss) / len(dc_loss)
        rgl_loss = sum(rgl_loss) / len(rgl_loss)
        mask_pred = torch.cat(mask_pred).to('cpu').numpy()
        mask_gt = torch.cat(mask_gt).to('cpu').numpy()
        image_batchs = torch.cat(image_batchs).to('cpu').numpy()
        label_batchs = torch.cat(label_batchs).to('cpu').numpy()
        image_pos_batchs = torch.cat(image_pos_batchs).to('cpu').numpy()
        image_id_batchs = np.array(image_id_batchs)
        logger.info(f'-----{split}-----')
        logger.info(
            f'Loss {epoch_loss:.10f}, ce_loss {ce_loss:.5f}, '
            f'dc_loss {dc_loss:.5f}, '
            f'rgl_loss {rgl_loss:.5f}, time cost {time_cost / 60:.2f} minutes')

        # calculate the metrics for the validation set
        mask_poseprocessed = mask_possprocess(mask_pred, 0.4)  # post-processing the predicted masks
        width = 3
        prec, rec, f1 = model_evaluation(label_batchs, mask_poseprocessed, image_id_batchs, image_pos_batchs, full_path_dict, label_radius=10, mask_ring=True, print_result=False, width=width)
        logger.info(f'width threshold {width:.5f}')
        logger.info(f'average precision {prec:.5f}, average recall {rec:.5f}, average f1 score {f1:.5f}')
        logger.info(f'average validating and testing time per image '
                f'{time_cost / (len(mask_gt) + len(mask_gt)) * 36}')
    if return_data:
        return (prec, rec,
                f1), mask_gt, mask_pred, mask_poseprocessed, image_batchs, label_batchs, image_pos_batchs, image_id_batchs
    return prec, rec, f1, epoch_loss


def main(config):
    global logger  # , writer
    datapath = config.dataset.datapath
    splitpath = config.dataset.splitpath

    # hyperparameters
    n_epoch = config.train.n_epoch
    lr = config.train.lr
    decay_weight = config.train.decay_weight
    decay_step = config.train.decay_step
    model_id = config.train.model_id
    start_epoch = config.test.start_epoch
    val_frequency = config.test.print_freqence
    batch_size = config.train.batch_size
    test_batch_size = config.test.batch_size
    subset = config.train.subset

    dataset_train = dataset_h5py(datapath, splitpath, split='train', subset=subset)
    dataset_test = dataset_h5py(datapath, splitpath, split='test', subset=subset)

    n_train = len(dataset_train)
    split = n_train // 20
    indices = list(range(n_train))
    random.shuffle(indices)
    sampler_train = torch.utils.data.sampler.SubsetRandomSampler(indices[split:])
    sampler_val = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])

    train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size, sampler=sampler_train,
                              collate_fn=collate_fn)
    val_loader = DataLoader(dataset=dataset_train, batch_size=test_batch_size, sampler=sampler_val,
                            collate_fn=collate_fn)
    test_loader = DataLoader(dataset=dataset_test, batch_size=test_batch_size, shuffle=False, collate_fn=collate_fn)

    if not os.path.exists('./logs/'):
        os.makedirs('./logs/')
    logger = Prepare_logger('./logs/', print_console=False)
    main_model = seg_model(config).to(device)

    if config.train.resume:
        load_model_path = f'../best_models/model_{model_id}.pth.tar'
        best_model = torch.load(load_model_path, map_location=device)
        main_model.load_state_dict(best_model)

    learned_params = filter(lambda p: p.requires_grad, main_model.parameters())
    optimizer = torch.optim.Adam(learned_params, lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_step, gamma=decay_weight)
    logger.info(
        f'unet model with, batch size {batch_size}, num_epochs {n_epoch}, initial learning rate {lr:.4f}')
    start = time.time()
    best_score = 0.35
    best_model = None
    with open(config.dataset.fullpath, 'r') as fp:
        full_path_dict = json.load(fp)
    for epoch in range(n_epoch):
        epoch_loss = train_epoch(epoch, main_model, train_loader, optimizer)
        if (epoch >= start_epoch) and (epoch % val_frequency == 0):
            prec, rec, f1, epoch_loss = evaluate_epoch(epoch, main_model, val_loader, 'validation', full_path_dict)
            prec, rec, f1, epoch_loss = evaluate_epoch(epoch, main_model, test_loader, 'test', full_path_dict)
            if f1 > best_score:
                logger.info(f'previous best f1: {best_score:4f}, '
                            f'current best f1 -- {f1:4f}')
                best_score = f1
                best_model = copy.deepcopy(main_model.state_dict())
        scheduler.step()
    if config.train.save_best:
        torch.save(best_model, f'../best_models/model_f1{best_score:.5f}.pth.tar')
    end = time.time()
    logger.info(f'total time cost: {(end - start) / 60:.2f} minutes')

    
def test(config):
    global logger
    if not os.path.exists('./logs/'):
        os.makedirs('./logs/')
    logger = Prepare_logger('./logs/', print_console=False)

    datapath = config.dataset.datapath
    splitpath = config.dataset.splitpath
    test_batch_size = config.test.batch_size
    load_model_path = f'../best_models/model_f10.86392.pth.tar'
    best_model = torch.load(load_model_path, map_location=device)

    main_model = seg_model(config).to(device)
    main_model.load_state_dict(best_model)
    dataset_test = dataset_h5py(datapath, splitpath, split='test')
    test_loader = DataLoader(dataset=dataset_test, batch_size=test_batch_size, shuffle=False, collate_fn=collate_fn)

    with open(config.dataset.fullpath, 'r') as fp:
        full_path_dict = json.load(fp)

    main_model.eval()
    with torch.no_grad():
        mask_pred = []
        mask_gt = []
        label_batchs = []
        image_pos_batchs = []
        image_id_batchs = []
        start = time.time()
        for batch_idx, batch_data in tqdm(enumerate(test_loader), desc='test'):
            image_batch, label_batch, image_pos, image_id = batch_data
            image_batch, label_batch = image_batch.to(device), label_batch.to(device)
            out_feature, loss = main_model(image_batch, label_batch)
            label_batchs.append(label_batch)
            image_pos_batchs.append(image_pos)
            image_id_batchs.extend(image_id)
            mask_pred.append(out_feature.sigmoid())
            mask_gt.append(label_batch)
        end = time.time()
        time_cost = end - start
        mask_pred = torch.cat(mask_pred).to('cpu').numpy()
        mask_gt = torch.cat(mask_gt).to('cpu').numpy()
        label_batchs = torch.cat(label_batchs).to('cpu').numpy()
        image_pos_batchs = torch.cat(image_pos_batchs).to('cpu').numpy()
        image_id_batchs = np.array(image_id_batchs)
        logger.info(f'-----test-----')
        # calculate the metrics for the test set
        mask_poseprocessed = mask_possprocess(mask_pred, 0.4)  # post-processing the predicted masks
        prec, rec, f1 = model_evaluation(label_batchs, mask_poseprocessed, image_id_batchs, image_pos_batchs, full_path_dict, label_radius=10, mask_ring=True, print_result=False)
        logger.info(f'average precision {prec:.5f}, average recall {rec:.5f}, average f1 score {f1:.5f}')
        logger.info(f'average validating and testing time per image '
                    f'{time_cost / (len(mask_gt) + len(mask_gt)) * 36}')
#     f1 = score[2]
#     dirname = './experiment/bnl_unet_july_full'
#     while os.path.exists(dirname):
#         dirname = dirname + '_'
#     dirname = dirname + f'_{f1:.5f}'
#     os.makedirs(dirname)
#     np.save(dirname + '/test_ids', im_id_test)
#     np.save(dirname + '/test_pos_ids', im_pos_test)
#     new_imagek = nib.Nifti1Image(image_batchs.astype(np.float64), affine=np.eye(4))
#     nib.save(new_imagek, dirname + '/testimage_save.nii')
#     new_imagek = nib.Nifti1Image(test_lbs.astype(np.float64), affine=np.eye(4))
#     nib.save(new_imagek, dirname + '/testlables_save.nii')
#     new_imagek = nib.Nifti1Image(mask_pred.astype(np.float64), affine=np.eye(4))
#     nib.save(new_imagek, dirname + '/testsoftnax_save.nii')
#     new_imagek = nib.Nifti1Image(mask_poseprocessed.astype(np.float64),
#                                  affine=np.eye(4))
#     nib.save(new_imagek, dirname + '/vsoftmax_processed.nii')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train or test?')
    parser.add_argument('mode', type=str, help='train or test?', nargs='?', default='train')

    args = parser.parse_args()

    if args.mode == 'train':
        main(config=config)
    else:
        test(config)


