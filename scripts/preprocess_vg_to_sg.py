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

with open(os.path.join(args.base_dir, 'annotations/captions_{}2017.json'.format(args.split)), 'r') as f_ann:
    coco_anns = json.load(f_ann)

def get_caption(num_rel, rel_objs, rel_sbjs, rel_preds, obj_names):
    count_name = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
                    'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'AB', 'AC', 'AD', 'AE', 'AF']
    
    all_obj_name_to_all_ids = dict()
    for i in range(num_rel):
        sub_id = rel_sbjs[i]
        obj_id = rel_objs[i]
        sub_name = vocab['object_idx_to_name'][obj_names[sub_id]]
        obj_name = vocab['object_idx_to_name'][obj_names[obj_id]]
        try:
            if sub_id not in all_obj_name_to_all_ids[sub_name]:
                all_obj_name_to_all_ids[sub_name].append(sub_id)
        except:
            all_obj_name_to_all_ids[sub_name] = [sub_id]
        try:
            if obj_id not in all_obj_name_to_all_ids[obj_name]:
                all_obj_name_to_all_ids[obj_name].append(obj_id)
        except:
            all_obj_name_to_all_ids[obj_name] = [obj_id]

    caption = []
    for i in range(num_rel):
        sbj_name = vocab['object_idx_to_name'][obj_names[rel_sbjs[i]]]
        obj_name = vocab['object_idx_to_name'][obj_names[rel_objs[i]]]
        pred_name = vocab['pred_idx_to_name'][rel_preds[i]]
        caption.append(sbj_name)
        if len(all_obj_name_to_all_ids[sbj_name]) > 1:
            caption.append(count_name[all_obj_name_to_all_ids[sbj_name].index(rel_sbjs[i])])
        caption.append(pred_name)
        caption.append(obj_name)
        if len(all_obj_name_to_all_ids[obj_name]) > 1:
            caption.append(count_name[all_obj_name_to_all_ids[obj_name].index(rel_objs[i])])
        caption.append(',')
        
    caption = caption[:-1]
    return ' '.join(caption)

from tqdm import tqdm

vg_anns = dict()

vg_anns['info'] = coco_anns['info']
vg_anns['licenses'] = coco_anns['licenses']

# split vg into train and valid

img_ids_check = {ii:1 for ii in f['image_ids'][...]}

img_output_anns = []
for img_ann in tqdm(vg_imgs):
    img_ann_single = dict()
    img_ann_single['license'] = 0
    img_ann_single['file_name'] = img_ann['url'].split('/')[-1]
    img_ann_single['coco_url'] = img_ann['url']
    img_ann_single['height'] = float(img_ann['height'])
    img_ann_single['width'] = float(img_ann['width'])
    img_ann_single['date_captured'] = '2013-11-14 11:18:45' 
    img_ann_single['flickr_url'] = img_ann['url']
    img_ann_single['id'] = int(img_ann['image_id'])
    try:
        tmp = img_ids_check[img_ann_single['id']]
        img_output_anns.append(img_ann_single)
    except:
        pass

sg_output_anns = []
count = 0
for idx, (img_id, rel_ids, num_rel, rel_objs, rel_sbjs, rel_preds, obj_names) in enumerate(tqdm(zip(f['image_ids'][...], f['relationship_ids'][...], 
                            f['relationships_per_image'][...], f['relationship_objects'][...], f['relationship_subjects'][...], 
                            f['relationship_predicates'][...], f['object_names'][...]))):
    annotation_ann_single = dict()
    annotation_ann_single['image_id'] = int(img_id)
    annotation_ann_single['id'] = int(img_id)
    annotation_ann_single['caption'] = get_caption(num_rel, rel_objs, rel_sbjs, rel_preds, obj_names)
    sg_output_anns.append(annotation_ann_single)
        
print('Done')
print('Num of image: ', len(img_output_anns))
print('Num of anns: ', len(sg_output_anns))

vg_anns['images'] = img_output_anns
vg_anns['annotations'] = sg_output_anns

# save json file
save_path = os.path.join(args.base_dir, '{}_sg.json'.format(args.split))
with open(save_path, 'w') as fp:
    json.dump(vg_anns, fp)