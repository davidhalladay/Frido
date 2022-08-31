import json
from itertools import chain
from pathlib import Path
from typing import Iterable, Dict, List, Callable, Any
from collections import defaultdict
import random
from csv import DictReader, reader as TupleReader
import os
import pickle

from tqdm import tqdm

from taming.data.annotated_objects_dataset import AnnotatedObjectsDataset
from taming.data.helper_types import Annotation, ImageDescription, Category
from taming.data.image_transforms import CenterCropReturnCoordinates, RandomCrop1dReturnCoordinates, \
    Random2dCropReturnCoordinates, RandomHorizontalFlipReturn, convert_pil_to_tensor


COCO_PATH_STRUCTURE = {
    'train': {
        'top_level': '',
        'instances_annotations': 'annotations/instances_train2017.json',
        'stuff_annotations': 'annotations/stuff_train2017.json',
        'files': 'train2017'
    },
    'validation': {
        'top_level': '',
        'instances_annotations': 'annotations/instances_val2017.json',
        'stuff_annotations': 'annotations/stuff_val2017.json',
        'files': 'val2017'
    }
}


COCO_PATH_STRUCTURE_14 = {
    'train': {
        'top_level': '',
        'instances_annotations': 'annotations/instances_train2014.json',
        'files': 'train2014'
    },
    'validation': {
        'top_level': '',
        'instances_annotations': 'annotations/instances_val2014.json',
        'files': 'val2014'
    }
}


def load_image_descriptions(description_json: List[Dict]) -> Dict[str, ImageDescription]:
    return {
        str(img['id']): ImageDescription(
            id=img['id'],
            license=img.get('license'),
            file_name=img['file_name'],
            coco_url=img['coco_url'],
            original_size=(img['width'], img['height']),
            date_captured=img.get('date_captured'),
            flickr_url=img.get('flickr_url')
        )
        for img in description_json
    }


def load_categories(category_json: Iterable) -> Dict[str, Category]:
    return {str(cat['id']): Category(id=str(cat['id']), super_category=cat['supercategory'], name=cat['name'])
            for cat in category_json if cat['name'] != 'other'}


def load_annotations(annotations_json: List[Dict], image_descriptions: Dict[str, ImageDescription],
                     category_no_for_id: Callable[[str], int], split: str, COCO_to_OI_cate_id=None) -> Dict[str, List[Annotation]]:
    annotations = defaultdict(list)
    total = sum(len(a) for a in annotations_json)
    for ann in tqdm(chain(*annotations_json), f'Loading {split} annotations', total=total):
        image_id = str(ann['image_id'])
        if image_id not in image_descriptions:
            raise ValueError(f'image_id [{image_id}] has no image description.')
        category_id = ann['category_id']
        if COCO_to_OI_cate_id is not None:
            try:
                category_id = COCO_to_OI_cate_id[str(category_id)]
            except:
                category_id = category_id
        try:
            category_no = category_no_for_id(str(category_id))
        except KeyError:
            continue

        width, height = image_descriptions[image_id].original_size
        bbox = (ann['bbox'][0] / width, ann['bbox'][1] / height, ann['bbox'][2] / width, ann['bbox'][3] / height)

        annotations[image_id].append(
            Annotation(
                id=ann['id'],
                area=bbox[2]*bbox[3],  # use bbox area
                is_group_of=ann['iscrowd'],
                image_id=ann['image_id'],
                bbox=bbox,
                category_id=str(category_id),
                category_no=category_no
            )
        )
    return dict(annotations)

def load_categories_OI(csv_path: Path) -> Dict[str, Category]:
    with open(csv_path) as file:
        reader = TupleReader(file)
        return {row[0]: Category(id=row[0], name=row[1], super_category=None) for row in reader}



class AnnotatedObjectsCoco(AnnotatedObjectsDataset):
    def __init__(self, use_things: bool = True, use_stuff: bool = True, img_id_file=None, caption_ann_path=None, 
                stuff_only=False, OI_cate_path='', 
                specific_img_ids=[], num_sample=-1, **kwargs):
        """
        @param data_path: is the path to the following folder structure:
                          coco/
                          ├── annotations
                          │   ├── instances_train2017.json
                          │   ├── instances_val2017.json
                          │   ├── stuff_train2017.json
                          │   └── stuff_val2017.json
                          ├── train2017
                          │   ├── 000000000009.jpg
                          │   ├── 000000000025.jpg
                          │   └── ...
                          ├── val2017
                          │   ├── 000000000139.jpg
                          │   ├── 000000000285.jpg
                          │   └── ...
        @param: split: one of 'train' or 'validation'
        @param: desired image size (give square images)
        """
        super().__init__(**kwargs)
        self.use_things = use_things
        self.use_stuff = use_stuff
        self.caption_ann_path = caption_ann_path
        self.OI_cate_path = OI_cate_path

        if stuff_only and self.paths['stuff_annotations'] is None:
            print('WARNING: Got stuff_only=True but stuff_json=None.')
            print('Falling back to stuff_only=False.')

        with open(self.paths['instances_annotations']) as f:
            inst_data_json = json.load(f)

        if use_stuff:
            with open(self.paths['stuff_annotations']) as f:
                stuff_data_json = json.load(f)

        if caption_ann_path is not None:
            with open(caption_ann_path) as f:
                caption_data_json = json.load(f)
            self.setup_caption(caption_data_json)

        img_id_used = dict()
        if img_id_file is not None:
            print('Load image ids file.')
            img_id_used = dict()
            with open(img_id_file) as file:
                lines = file.readlines()
                if num_sample != -1:
                    lines = lines[:num_sample]
                    print('load only {} images.'.format(num_sample))
                for line in lines:
                    img_id_used[line.rstrip()] = 1

        category_jsons = []
        annotation_jsons = []
        if self.use_things:
            category_jsons.append(inst_data_json['categories'])
            annotation_jsons.append(inst_data_json['annotations'])
        if self.use_stuff:
            category_jsons.append(stuff_data_json['categories'])
            annotation_jsons.append(stuff_data_json['annotations'])

        # image id contains stuff
        if stuff_only:
            image_ids_with_stuff = dict()
            for img_ann in stuff_data_json['annotations']:
                try:
                    image_ids_with_stuff[str(img_ann['image_id'])] += 1
                except:
                    image_ids_with_stuff[str(img_ann['image_id'])] = 1

        self.categories = load_categories(chain(*category_jsons))
        COCO_to_OI_cate_id = None
        if OI_cate_path != '':
            self.categories_OI = load_categories_OI(OI_cate_path)
            oi_names_raw = ['-'.join(v.name.lower().split(' ')) for k, v in self.categories_OI.items()]
            oi_names_raw += [v.name.lower() for k, v in self.categories_OI.items()]
            oi_names = [v.name for k, v in self.categories_OI.items()] * 2
            oi_ids = [k for k, v in self.categories_OI.items()] * 2
            COCO_to_OI_cate_name = dict()
            COCO_to_OI_cate_id = dict()
            self.categories_append = dict()
            for k, v in self.categories.items():
                if v.name not in oi_names_raw:
                    self.categories_append[k] = v
                else:
                    COCO_to_OI_cate_name[v.name] = oi_names[oi_names_raw.index(v.name)]
                    COCO_to_OI_cate_id[k] = oi_ids[oi_names_raw.index(v.name)]
            self.categories = self.categories_OI
        
        self.filter_categories()
        self.setup_category_id_and_number()
        self.image_descriptions = load_image_descriptions(inst_data_json['images'])
        annotations = load_annotations(annotation_jsons, self.image_descriptions, 
                                        self.get_category_number, self.split, COCO_to_OI_cate_id)
        self.annotations = self.filter_object_number(annotations, self.min_object_area,
                                                     self.min_objects_per_image, self.max_objects_per_image)
        self.image_ids = sorted(list(self.annotations.keys()))
        if stuff_only:
            new_image_ids = []
            for image_id in self.image_ids:
                try:
                    exist = image_ids_with_stuff[image_id]
                    if exist: new_image_ids.append(image_id)
                except:
                    print("Adopt stuff data only!")
            self.image_ids = new_image_ids

        if len(img_id_used) > 0:
            new_image_ids = []
            for image_id in self.image_ids:
                try:
                    exist = img_id_used['{:012d}'.format(int(image_id))]
                    if exist: new_image_ids.append(image_id)
                except:
                    pass
            self.image_ids = new_image_ids

        if caption_ann_path is not None:
            cap_image_ids = sorted(list(self.img_id_to_caption_list.keys()))
            self.image_ids = sorted(list(set(self.image_ids).intersection(set(cap_image_ids))))

        self.clean_up_annotations_and_image_descriptions()

        if len(specific_img_ids) != 0:
            specific_img_ids_tmp = dict()
            for ii in specific_img_ids:
                specific_img_ids_tmp[ii] = 1
            specific_img_ids = specific_img_ids_tmp

            print('Detect specific image id specified:', len(specific_img_ids))
            tmp_img_ids = []
            for ii in self.image_ids:
                try:
                    _ = specific_img_ids[ii]
                    tmp_img_ids.append(ii)
                except:
                    pass

            self.image_ids = tmp_img_ids

    def setup_caption(self, caption_data_json):
        img_id_to_caption_list = dict()
        for ann in caption_data_json['annotations']:
            file_caption = ann['caption'].replace('.', '')
            try:
                img_id_to_caption_list[str(ann['image_id'])].append(file_caption)
            except:
                img_id_to_caption_list[str(ann['image_id'])] = [file_caption]
        self.img_id_to_caption_list = img_id_to_caption_list

    def get_path_structure(self) -> Dict[str, str]:
        if self.split not in COCO_PATH_STRUCTURE:
            raise ValueError(f'Split [{self.split} does not exist for COCO data.]')
        if '2017' in self.data_path:
            return COCO_PATH_STRUCTURE[self.split]
        elif '2014' in self.data_path:
            return COCO_PATH_STRUCTURE_14[self.split]
        else:
            raise ValueError('Incorrect data structure.')

    def get_image_path(self, image_id: str) -> Path:
        return self.paths['files'].joinpath(self.image_descriptions[str(image_id)].file_name)

    def get_image_description(self, image_id: str) -> Dict[str, Any]:
        # noinspection PyProtectedMember
        return self.image_descriptions[image_id]._asdict()

    def get_image_caption(self, image_id: str) -> List[str]:
        # noinspection PyProtectedMember
        return self.img_id_to_caption_list[image_id]

    def __getitem__(self, n: int) -> Dict[str, Any]:
        image_id = self.get_image_id(n)
        sample = self.get_image_description(image_id)
        sample['annotations'] = self.get_annotation(image_id)

        if 'image' in self.keys:
            sample['image_path'] = str(self.get_image_path(image_id))
            sample['image'] = self.load_image_from_disk(sample['image_path'])
            sample['image'] = convert_pil_to_tensor(sample['image'])
            sample['crop_bbox'], sample['flipped'], sample['image'] = self.image_transform(sample['image'])
            sample['image'] = sample['image'].permute(1, 2, 0)

        if self.caption_ann_path is not None:
            sample['caption'] = self.get_image_caption(image_id)[0]

        for conditional, builder in self.conditional_builders.items():
            if conditional in self.keys:
                sample[conditional] = builder.build(sample['annotations'], sample['crop_bbox'], sample['flipped'])

        if self.keys:
            sample = {key: sample[key] for key in self.keys}
        return sample