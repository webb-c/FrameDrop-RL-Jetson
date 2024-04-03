import csv
import json
import logging
from pathlib import Path

import yaml

from src.Reducto.util.utils import assert_list


def load_yaml(filepath):
    filepath = Path(filepath)
    if not filepath.exists():
        logging.warning(f'load_yaml: file not exists: {filepath}')
    with open(filepath, 'r') as yf:
        try:
            file = yaml.safe_load(yf)
        except yaml.YAMLError as ex:
            logging.warning(ex)
    return file


def load_json(path):
    """주어진 path에 존재하는 json파일을 읽어서 반환한다. 없으면 None을 반환한다. 
    Args:
        path (str): 읽어올 json파일의 경로
    Returns:
        Dict: json 파일에 저장된 내용
    """
    if path is None or not Path(path).exists():
        assert print(path, "Path is Not Found")
        
    with open(path, 'r') as j:
        data = json.load(j)
    return data


def dump_json(data, path, mkdir=False):
    """주어진 path에 data를 json파일의 형태로 쓴다.
    Args:
        data (OrderedDict): json파일에 쓰고자하는 데이터
        path (str): 작성할 json파일의 경로
        mkdir (bool, optional): json파일 경로에 있는 directory가 존재하지 않을 때 만들 것인자. Defaults to False.
    """
    if mkdir and not Path(path).parent.exists():
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as j:
        json.dump(data, j)


def load_csv(path):
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        res = {key: [] for key in reader.fieldnames}
        for row in reader:
            for k, v in dict(row).items():
                res[k].append(v)
    return res


def dump_csv(data, path):
    with open(path, 'w') as f:
        f.write(','.join(list(data[0].keys())) + '\n')
        for entry in data:
            vs = [str(v) for _, v in entry.items()]
            f.write(','.join(vs))
            f.write('\n')


def load_evaluation(path, differ, metrics):
    evl_raw = load_json(path)
    metrics = assert_list(metrics, str)

    evl_differs = evl_raw.keys()
    assert differ in evl_differs, f'{differ} not exists in {path}'

    evl_formatted = {
        differ: {
            float(thresh): {
                metric: abs(score)
                for metric, score in evl_raw[differ][thresh].items()
                if metric in metrics
            }
            for thresh in evl_raw[differ]
        }
    }

    return evl_formatted


def load_diff_vector(path, differ):
    diff_raw = load_json(path)

    diff_differs = diff_raw.keys()
    assert differ in diff_differs, f'{differ} not exists in {path}'

    diff_vectors = {
        differ: diff_raw[differ]['diff_vector'],
    }

    return diff_vectors


def load_diff_result(path, differ):
    diff_raw = load_json(path)

    diff_differs = diff_raw.keys()
    assert differ in diff_differs, f'{differ} not exists in {path}'

    diff_results = {
        differ: {
            float(thresh): res
            for thresh, res in diff_raw[differ]['result'].items()
        }
    }

    return diff_results


def load_inference(path):
    """주어진 path에 존재하는 json 파일을 읽어서 반환한다.
    Args:
        path (str): 읽어올 파일의 경로
    Calls:
        load_json()
    Returns:
        Dict: json파일의 내용
    """
    inference_raw = load_json(path)
    return inference_raw


def load_motion(path):
    motion_raw = load_json(path)
    return motion_raw
