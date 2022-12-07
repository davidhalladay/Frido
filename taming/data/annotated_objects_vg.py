import json
from itertools import chain
from pathlib import Path
from typing import Iterable, Dict, List, Callable, Any
from collections import defaultdict
import random

from tqdm import tqdm

from taming.data.annotated_objects_dataset import AnnotatedObjectsDataset
from taming.data.helper_types import Annotation, ImageDescription, Category
from taming.data.image_transforms import CenterCropReturnCoordinates, RandomCrop1dReturnCoordinates, \
    Random2dCropReturnCoordinates, RandomHorizontalFlipReturn, convert_pil_to_tensor

VG_PATH_STRUCTURE = {
    'train': {
        'top_level': '',
        'image_data': 'image_data.json',
        'files': 'VG_100K'
    },
    'validation': {
        'top_level': '',
        'image_data': 'image_data.json',
        'files': 'VG_100K'
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


class AnnotatedObjectsVg(AnnotatedObjectsDataset):
    def __init__(self, use_things: bool = True, use_stuff: bool = True, caption_ann_path=None, specific_img_ids=[], **kwargs):

        super().__init__(**kwargs)
        self.caption_ann_path = caption_ann_path

        with open(caption_ann_path) as f:
            caption_data_json = json.load(f)
        self.setup_caption(caption_data_json)

        image_ids = [str(img_ann['id']) for img_ann in caption_data_json['images']]

        self.image_descriptions = load_image_descriptions(caption_data_json['images'])

        self.image_ids = sorted(image_ids)

        if len(specific_img_ids) != 0:
            print('Detect specific image id specified:', specific_img_ids)
            tmp_img_ids = []
            for s_ii in specific_img_ids:
                for ii in self.image_ids:
                    if s_ii in ii:
                        tmp_img_ids.append(ii)
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
        if self.split not in VG_PATH_STRUCTURE:
            raise ValueError(f'Split [{self.split} does not exist for COCO data.]')
        try:
            return VG_PATH_STRUCTURE[self.split]
        except:
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

        if 'image' in self.keys:
            sample['image_path'] = str(self.get_image_path(image_id))
            sample['image'] = self.load_image_from_disk(sample['image_path'])
            sample['image'] = convert_pil_to_tensor(sample['image'])
            sample['crop_bbox'], sample['flipped'], sample['image'] = self.image_transform(sample['image'])
            sample['image'] = sample['image'].permute(1, 2, 0)

        if self.caption_ann_path is not None:
            sample['caption'] = random.choice(self.get_image_caption(image_id))

        if self.keys:
            # only return specified keys
            sample = {key: sample[key] for key in self.keys}
        return sample