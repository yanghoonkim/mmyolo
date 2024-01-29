from pathlib import Path
import pandas as pd
import json
    
from nia.nia_dataset_reader import (
    NiaDataPathExtractor,
    DataFrameSplitter,
    NiaDataPathProvider,
)

img_prefix = '1.원천데이터'
data_root = '/root/mmyolo/data/nia/'
train_ann_file = 'thermal_train_label.json'
val_ann_file = 'thermal_valid_label.json'
test_ann_file = 'thermal_test_label.json'

BASE_PATH = Path(data_root)
IMG_PATH = BASE_PATH /  img_prefix
TRAIN_LABEL_PATH = BASE_PATH / train_ann_file
VALID_LABEL_PATH = BASE_PATH / val_ann_file
TEST_LABEL_PATH = BASE_PATH / test_ann_file

# categoires 사전 정의
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

def to_frame(pairs):
    df = pd.DataFrame(pairs, columns=['imgpath', 'annopath'])
    df.index = df.imgpath.apply(lambda x: x.split('/')[-1])
    df.index.name = 'filename'
    return df

# thermal 데이터 오류 검출
# 열영상 데이터 annotations에서 category_id가 [3, 2, 8, 1] 범주를 초과한 경우들 제외하기
# json 파일에서 필요한 정보만 추출: 'images', 'annotations'
def make_dict(df):
    anno_images = list()
    anno_annotations = list()

    for filename, item in zip(df.imgpath, df.annopath):
        issue_flag = False
        with open(item) as f:
            item_json = json.load(f)
        for anno in item_json['annotations']:
            if anno['category_id'] not in [3, 2, 8, 1]:
                issue_flag = True
                break
        if not issue_flag:
            anno_images.extend(item_json['images'])
            anno_images[-1]['file_name'] = Path(filename).relative_to(IMG_PATH.as_posix()).as_posix()
            anno_annotations.extend(item_json['annotations'])
    
    dict_ = {'categories': categories, 'images': anno_images, 'annotations': anno_annotations}

    return dict_

def split_data():
    if (not TRAIN_LABEL_PATH.exists()) or (not VALID_LABEL_PATH.exists()) or (not TEST_LABEL_PATH.exists()):
        print('[DATA SPLIT] Splitting data...')

        path_provider = NiaDataPathProvider(
            reader=NiaDataPathExtractor(dataset_dir=data_root),
        )        

        train_path_pairs = path_provider.get_split_data_list(channels="thermal", splits="train")
        valid_path_pairs = path_provider.get_split_data_list(channels="thermal", splits='valid')
        test_path_pairs = path_provider.get_split_data_list(channels="thermal", splits='test')

        df_thermal_train = to_frame(train_path_pairs)
        df_thermal_valid = to_frame(valid_path_pairs)
        df_thermal_test = to_frame(test_path_pairs)

        train_dict = make_dict(df_thermal_train)
        valid_dict = make_dict(df_thermal_valid)
        test_dict = make_dict(df_thermal_test)

        # annotation id 중복 이슈 해결
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


        with TRAIN_LABEL_PATH.open('w') as f:
            json.dump(train_dict, f)
        
        with VALID_LABEL_PATH.open('w') as f:
            json.dump(valid_dict, f)

        with TEST_LABEL_PATH.open('w') as f:
            json.dump(test_dict, f)         

    else:
        print('[DATA SPLIT] Load existing files...')    