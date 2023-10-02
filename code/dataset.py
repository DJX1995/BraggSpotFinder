import numpy as np
import h5py
import json
import os
from util import *
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from data_preprocess import parse_header, fix_header


class dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, images, labels, im_id, im_pos):
        """
        Args:
            vname_file (string): Path to the video name file
            word2idx_path
            caption_file (string): Path to the caption file with timestamp annotations.
            v_feat_path (string): Directory with all c3d or i3d files.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.images = images
        self.labels = labels
        self.im_id = im_id
        self.im_pos = im_pos
        self.w = 6.
        self.h = 6.

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        patch = self.images[index]
        label = self.labels[index]
        patch_id = self.im_id[index]
        patch_pos = self.im_pos[index]
        patch = torch.from_numpy(patch)
        label = torch.from_numpy(label)
        patch_pos = torch.from_numpy(patch_pos)
        return patch, label, patch_id, patch_pos

    
class dataset_h5py(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, datapath, splitpath, im_ids=None, split='train', subset=None):
        """
        Args:
            vname_file (string): Path to the video name file
            word2idx_path
            caption_file (string): Path to the caption file with timestamp annotations.
            v_feat_path (string): Directory with all c3d or i3d files.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.split = split
        if im_ids != None:
            self.im_id_to_read = im_ids
        else:
            with open(splitpath, 'r') as fp:
                vnames = json.load(fp)
            if subset != None:
                self.im_id_to_read = [sample.split('/')[-1] for sample in vnames[split]][:subset]
            else:
                self.im_id_to_read = [sample.split('/')[-1] for sample in vnames[split]]
        self.process_data(datapath, subset)
        self.w = 6.
        self.h = 6.

    def process_data(self, datapath, subset):
        images = []
        labels = []
        im_id = []
        im_pos = []
        for idx, imid in tqdm(enumerate(self.im_id_to_read), total=len(self.im_id_to_read), desc=f'loading {self.split} data'):
            try:
                with h5py.File(datapath, 'r') as hf:
                    images.append(hf[imid]['im'][()])
                    labels.append(hf[imid]['lbs'][()])
                    im_pos.append(hf[imid]['im_pos'][()])
                    num = hf[imid]['im_pos'][()].shape[0]
                    im_id.append([imid] * num)
            except:  # if there is a sample that does not have any annotations, skip it
                continue
        self.images = np.concatenate(images, axis=0)
        self.labels = np.concatenate(labels, axis=0)
        self.im_pos = np.concatenate(im_pos, axis=0)
        self.im_id = np.concatenate(im_id, axis=0)
            
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        patch = self.images[index]
        label = self.labels[index]
        patch_id = self.im_id[index]
        patch_pos = self.im_pos[index]
        
        patch = torch.from_numpy(patch)
        label = torch.from_numpy(label)
        patch_pos = torch.from_numpy(patch_pos)
        return patch, label, patch_pos, patch_id


def collate_fn(batch):
    images, labels, patch_pos, patch_id = zip(*batch)
    images = torch.stack(images)
    labels = torch.stack(labels)
    patch_pos = torch.stack(patch_pos)
    patch_id = list(patch_id)
    return images.float(), labels, patch_pos, patch_id



if __name__ == "__main__":
    im, lbs = load_data_quicknpy()
    # split the dataset into training set, validation set and testing set
    val_im, val_lbs = im[int(len(im) * 0.8):int(len(im) * 0.85)], lbs[int(len(im) * 0.8):int(len(im) * 0.85)]
    test_im, test_lbs = im[int(len(im) * 0.85):], lbs[int(len(im) * 0.85):]
    im, lbs = im[:int(len(im) * 0.8)], lbs[:int(len(im) * 0.8)]
    dataset_train = dataset(im, lbs)
    dataset_val = dataset(val_im, val_lbs)
    dataset_test = dataset(test_im, test_lbs)
    batch_size = 8
    test_batch_size = 8
    train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(dataset=dataset_test, batch_size=test_batch_size, shuffle=True, collate_fn=collate_fn)
    for batch_idx, batch_data in tqdm(enumerate(train_loader), total=int(len(dataset_train) / batch_size),
                                      desc='generating batch data'):
        image_batch, label_batch = batch_data
        pass
    print('Done')