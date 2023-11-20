# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import os.path as osp

from mmdet.utils import setup_cache_size_limit_of_dynamo
from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import Runner

from mmyolo.registry import RUNNERS
from mmyolo.utils import is_metainfo_lower

from pathlib import Path
import json
import pandas as pd
import math
import numpy as np

SEED = 0
BASE_PATH = Path('/root/mmyolo/data/nia/')
ANNO_PATH = BASE_PATH / '라벨링데이터'
COLL_PATH = BASE_PATH / '원천데이터'
TRAIN_LABEL_PATH = BASE_PATH / 'thermal_train_label.json'
VALID_LABEL_PATH = BASE_PATH / 'thermal_valid_label.json'
TEST_LABEL_PATH = BASE_PATH / 'thermal_test_label.json'


# raw data에 오류가 있기 때문에 사전에 정의해줌
categories = [
 {'id': 3,
  'name': 'car-b',
  'category_id': '937f7e78-88b3-48ec-bb02-6f995a363973',
  'supercategory': 'BoundingBox'},
 {'id': 2,
  'name': 'Two-wheel Vehicle-b',
  'category_id': 'c439976a-a118-43a9-a2d3-9817be51fc21',
  'supercategory': 'BoundingBox'},
 {'id': 8,
  'name': 'TruckBus-b',
  'category_id': '7029be4d-1f5c-4c2f-acd9-5dea0e8f8e11',
  'supercategory': 'BoundingBox'},
 {'id': 1,
  'name': 'Pedestrian-b',
  'category_id': '30438bfd-8f97-4897-bd1a-7028b3384f33',
  'supercategory': 'BoundingBox'}]


def is_thermal_data(item):
    cond = item.match('thermal/*.png*')
    return cond

def split_data():
    if (not TRAIN_LABEL_PATH.exists()) or (not VALID_LABEL_PATH.exists()) or (not TEST_LABEL_PATH.exists()):
        print('[DATA SPLIT] Splitting data...')

        img_paths = list(COLL_PATH.rglob('*.png'))
        img_names= [item.name for item in img_paths]

        anno_paths = list(ANNO_PATH.rglob('*.json'))
        anno_names= [item.name for item in anno_paths]


        thermal_anno_paths = list(filter(is_thermal_data, anno_paths))
        thermal_anno_paths = list(filter(lambda x: '._' not in x.as_posix(), thermal_anno_paths))
        thermal_anno_names = [item.name for item in thermal_anno_paths]
        thermal_anno_names_wo_json = [item.rstrip('.json') for item in thermal_anno_names]

        thermal_img_paths = list(filter(is_thermal_data, img_paths))
        thermal_img_names = [item.name for item in thermal_img_paths]    

        df_thermal_img = pd.DataFrame({'filename': thermal_img_names, 'imgpath': thermal_img_paths}).set_index('filename')
        df_thermal_anno = pd.DataFrame({'filename': thermal_anno_names_wo_json, 'annopath': thermal_anno_paths}).set_index('filename')

        df_thermal = pd.concat([df_thermal_img, df_thermal_anno], axis=1).dropna(how='any')
        df_thermal = df_thermal.sample(frac=1, random_state=SEED) # random shuffle

        # thermal 데이터 오류 검출
        # 열영상 데이터 annotations에서 category_id가 [3, 2, 8, 1] 범주를 초과한 경우들 제외하기
        # json 파일에서 필요한 정보만 추출: 'images', 'annotations'
        filenames = list()
        anno_images = list()
        anno_annotations = list()

        for item in df_thermal.annopath:
            issue_flag = False
            item_json = json.load(item.open())
            for anno in item_json['annotations']:
                if anno['category_id'] not in [3, 2, 8, 1]:
                    issue_flag = True
                    break
            if not issue_flag:
                filenames.append(item.name.rstrip('.json'))
                anno_images.append(item_json['images'])
                anno_annotations.append(item_json['annotations'])

                
        # 'images'의 file_name이 이상한 경우가 있어서 바꿔주기
        for i, item in enumerate(anno_images):
            anno_images[i][0]['file_name'] = filenames[i]

        ratio = [8, 1, 1] # train / valid / test
        ratio = [item / sum(ratio) for item in ratio]

        total_len = len(anno_images)
        train_len = math.floor(total_len * ratio[0])
        valid_len = train_len + math.floor(total_len * ratio[1])

        train_dict = dict()
        valid_dict = dict()
        test_dict = dict()

        train_dict['categories'] = categories
        train_dict['images'] = anno_images[:train_len]
        train_dict['images'] = np.concatenate(train_dict['images']).tolist() # [[1],[2],[3]] -> [1,2,3]
        train_dict['annotations'] = anno_annotations[:train_len] 
        train_dict['annotations'] = np.concatenate(train_dict['annotations']).tolist() # [[1,2],[3,4],[5,6]] -> [1,2,3,4,5,6]

        valid_dict['categories'] = categories
        valid_dict['images'] = anno_images[train_len:valid_len]
        valid_dict['images'] = np.concatenate(valid_dict['images']).tolist()
        valid_dict['annotations'] = anno_annotations[train_len:valid_len]
        valid_dict['annotations'] = np.concatenate(valid_dict['annotations']).tolist()

        test_dict['categories'] = categories
        test_dict['images'] = anno_images[valid_len:]
        test_dict['images'] = np.concatenate(test_dict['images']).tolist()
        test_dict['annotations'] = anno_annotations[valid_len:]
        test_dict['annotations'] = np.concatenate(test_dict['annotations']).tolist()


        # annotation id 중복 이슈 해결
        temp = set()
        anno_id = 0
        for idx, item in enumerate(train_dict['annotations']):
            train_dict['annotations'][idx]['id'] = anno_id
            anno_id += 1
        for idx, item in enumerate(valid_dict['annotations']):
            valid_dict['annotations'][idx]['id'] = anno_id
            anno_id += 1
        for idx, item in enumerate(test_dict['annotations']):
            test_dict['annotations'][idx]['id'] = anno_id
            anno_id += 1

        #### folder hierarchy가 다를수도 있기 때문에 실제 file_name으로 바꿔주기
        for idx, item in enumerate(train_dict['images']):
            file_name_wo_dir = item['file_name']
            real_file_name = df_thermal.loc[file_name_wo_dir, 'imgpath'].relative_to('/root/mmyolo/data/nia/원천데이터/').as_posix()
            train_dict['images'][idx]['file_name'] = real_file_name

        for idx, item in enumerate(valid_dict['images']):
            file_name_wo_dir = item['file_name']
            real_file_name = df_thermal.loc[file_name_wo_dir, 'imgpath'].relative_to('/root/mmyolo/data/nia/원천데이터/').as_posix()
            valid_dict['images'][idx]['file_name'] = real_file_name

        for idx, item in enumerate(test_dict['images']):
            file_name_wo_dir = item['file_name']
            real_file_name = df_thermal.loc[file_name_wo_dir, 'imgpath'].relative_to('/root/mmyolo/data/nia/원천데이터/').as_posix()
            test_dict['images'][idx]['file_name'] = real_file_name

        with TRAIN_LABEL_PATH.open('w') as f:
            json.dump(train_dict, f)
        
        with VALID_LABEL_PATH.open('w') as f:
            json.dump(valid_dict, f)

        with TEST_LABEL_PATH.open('w') as f:
            json.dump(test_dict, f)         

    else:
        print('[DATA SPLIT] Load existing files...')    



def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
        'specify, try to auto resume from the latest checkpoint '
        'in the work directory.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    split_data()
    args = parse_args()

    # Reduce the number of repeated compilations and improve
    # training speed.
    setup_cache_size_limit_of_dynamo()

    # load config
    cfg = Config.fromfile(args.config)
    # replace the ${key} with the value of cfg.key
    # cfg = replace_cfg_vals(cfg)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # resume is determined in this priority: resume from > auto_resume
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    # Determine whether the custom metainfo fields are all lowercase
    is_metainfo_lower(cfg)

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # start training
    runner.train()


if __name__ == '__main__':
    main()
