import os
import json

import argparse
import random
import shutil
from PIL import Image
from tqdm import tqdm
import torch
import numpy as np
import h5py

parser = argparse.ArgumentParser(description='Process visual genome.')
parser.add_argument("-b", '--base_dir', type=str,
                    help='path to raw visual genome')
parser.add_argument("-s", '--split', type=str,
                    help='train or val split')
args = parser.parse_args()


assert args.split == 'train' or args.split == 'val' 

# loading data

with open(os.path.join(args.base_dir, 'image_data.json'), 'r') as f:
    vg_imgs = json.load(f)

with open(os.path.join(args.base_dir, 'vocab.json'), 'r') as f:
    vocab = json.load(f)

f = h5py.File(os.path.join(args.base_dir, '{}.h5'.format(args.split)), 'r')

with open(os.path.join(args.base_dir, 'annotations', 'instances_{}2017.json'.format(args.split)), 'r') as f_ann:
    coco_anns = json.load(f_ann)


### collect visual genome data into coco annotation format.

vg_anns = coco_anns.copy()
vg_anns['images'] = []
vg_anns['annotations'] = []
vg_anns['categories'] = []

for idx, (name, idx) in enumerate(vocab['object_name_to_idx'].items()):
    single = {'supercategory': name, 'id': int(idx), 'name': name}
    vg_anns['categories'].append(single)

vg_img_id_to_info = dict()
for ann in vg_imgs:
    vg_img_id_to_info[ann['image_id']] = ann


for idx, (img_id, img_filename) in enumerate(tqdm(zip(f['image_ids'][...], f['image_paths'][...]))):
    img_filename = img_filename.decode('utf-8').split('/')[-1]
    ann = vg_img_id_to_info[img_id]
    single = {
        'license': 1,
        'file_name': img_filename,
        'coco_url': ann['url'],
        'height': int(ann['height']),
        'width': int(ann['width']),
        'date_captured': '2013-11-14 22:32:02',
        'flickr_url': ann['url'],
        'id': int(img_id)
    }
    vg_anns['images'].append(single)

for idx, (img_id, obj_ids, obj_cate_ids, obj_bboxes) in enumerate(tqdm(zip(f['image_ids'][...], f['object_ids'][...], f['object_names'][...], f['object_boxes'][...]))):
    for jj in range(len(obj_ids)):
        if obj_ids[jj] != -1:
            single = {'segmentation': [],
                'iscrowd': 0,
                'image_id': int(img_id),
                'bbox': list(obj_bboxes[jj].astype(np.float64)),
                'category_id': int(obj_cate_ids[jj]),
                'id': int(obj_ids[jj])}
            vg_anns['annotations'].append(single)

# save results
with open(os.path.join(args.base_dir, '{}_coco_style.json'.format(args.split)), 'w') as f:
    json.dump(vg_anns, f)