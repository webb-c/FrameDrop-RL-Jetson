import argparse
from pathlib import Path
import re
import os
from typing import Tuple, List, Dict, Union

import yaml
from utils.util import str2bool


def parse_args() : 
    parser = argparse.ArgumentParser()
    
    ### base
    parser.add_argument("-method", "--test_method", type=str, default=None, help="testing algorithm")
    parser.add_argument("-video", "--video_path", type=str, default=None, help="testing video path")
    parser.add_argument("-fps", "--fps", type=int, default=30, help="frame per sec")
    parser.add_argument("-model", "--model_path", type=str, default=None, help="trained model path")
    
    parser.add_argument("-write", "--write", type=str2bool, default=False, help="make output video?")
    parser.add_argument("-out", "--output_path", type=str, default=None, help="output video Path")
    parser.add_argument("-f1", "--f1_test", type=str2bool, default=False, help="showing f1 score")  # doing at Server
    
    parser.add_argument("-debug", "--debug_mode", type=str2bool, default=False, help="debug tool")
    parser.add_argument("-jetson", "--jetson_mode", type=str2bool, default=False, help="running in Jetson Nano")
    
    ### lrlo
    parser.add_argument("-V", "--V", type=float, default=None, help="trade off parameter between stability & accuracy")
    
    parser.add_argument("-mask", "--is_masking", type=str2bool, default=True, help="using lyapunov based guide?")
    parser.add_argument("-state", "--state_method", type=int, default=1, help="state define method")

    ### FrameHopper
    # parser.add_argument("-target", "--target_acc", type=float, default=None, help="target f1 score")
    
    ### Reducto
    parser.add_argument("-dist", "--distance", type=float, default=None, help="diff vector that which type used feature")
    parser.add_argument("-safe", "--safe_zone", type=float, default=None, help="testing metric")
    parser.add_argument("-target", "--target_acc", type=float, default=None, help="target f1 score")
    
    parser.add_argument("-s", "--subset", type=str, default='subset0', help="testing subset name in video dataset")
    parser.add_argument("-differ", "--differ", type=str, default="edge", help="diff vector that which type used feature")
    parser.add_argument("-metric", "--metric", type=str, default="mAP-all", help="testing metric")
    
    return parser.parse_args()


def parse_lrlo_test(conf:Dict[str, Union[str, int, bool, float]]) -> Dict[str, Union[str, int, bool, float]]:
    """모델 경로에 기록된 각종 정보를 통해 conf를 설정합니다.
    Args:
        conf (Dict[str, Union[str, int, bool, float]]): parse_args로 전달받은 기본 설정
    Returns:
        Dict[str, Union[str, int, bool, float]]: model_path parsing으로 분석한 설정이 추가된 dict
    """
    conf['state_num'] = 15
    conf['state_method'] = 1
    root_data = "./data/"
    root_cluster = "./model/LRLO/cluster/"

    model_path = conf['model_path']
    name = re.split(r"[/\\]", model_path)[-1][:-4]
    parts = name.split('_')
    cluster_video_name = ""
    
    for i in range(1, len(parts), 2):
        key = parts[i]
        value = parts[i+1]
        if key == 'videopath':
            cluster_video_name = value
        if key == 'actiondim':
            conf['action_dim'] = int(value)
        if key == 'radius':
            conf['radius'] = int(value) 

    conf['cluster_path'] = os.path.join(root_cluster, cluster_video_name + "_" + str(conf['state_num']) + "_" + str(conf['radius']) + "_" + str(conf['action_dim']) + "_" +  str(conf['state_method']) + ".pkl")
    conf['video_path'] = os.path.join(root_data, conf['video_name'] + ".mp4")
    
    return conf


def parse_frameHopper_test(conf:Dict[str, Union[str, int, bool, float]]) -> Dict[str, Union[str, int, bool, float]]:
    """모델 경로에 기록된 각종 정보를 통해 conf를 설정합니다.
    Args:
        conf (Dict[str, Union[str, int, bool, float]]): parse_args로 전달받은 기본 설정
    Returns:
        Dict[str, Union[str, int, bool, float]]: model_path parsing으로 분석한 설정이 추가된 dict
    """
    conf['state_num'] = 10
    root_data = "./data/"
    root_cluster = "./model/FrameHopper/cluster/"

    model_path = conf['model_path']
    name = re.split(r"[/\\]", model_path)[-1][:-4]
    parts = name.split('_')
    cluster_video_name = ""
    
    for i in range(1, len(parts), 2):
        key = parts[i]
        value = parts[i+1]
        if key == 'videopath':
            cluster_video_name = value

    conf['cluster_path'] = os.path.join(root_cluster, cluster_video_name + ".pkl")
    conf['video_path'] = os.path.join(root_data, conf['video_name'] + ".mp4")
    
    return conf


def parse_reducto_test(conf:Dict[str, Union[str, int, bool, float]]) -> Dict[str, Union[str, int, bool, float]]:
    """모델 경로에 기록된 각종 정보를 통해 conf를 설정합니다.

    Args:
        conf (Dict[str, Union[str, int, bool, float]]): parse_args로 전달받은 기본 설정

    Returns:
        Tuple[Dict[str, Union[str, int, bool, float]], str]: model_path parsing으로 분석한 설정이 추가된 dict, log_path
    """
    root_data = "./data/split/"
    root_cluster = "./model/Reducto/cluster/"
    root_config = "./model/Reducto/config/"
    dataset_mapping = {
        'JK-1' : 'JK',
        'SD-1' : 'SD',
        'JN-1' : 'JN'
    }
    conf['dataset'] = conf['video_name']
    conf['cluster_path'] = os.path.join(root_cluster, dataset_mapping[conf['dataset']] + "_" + str(conf['safe_zone']) + "_" + str(conf['target_acc']) + ".pkl")
    conf['config_path'] = os.path.join(root_config, conf['video_name'] + ".yaml")
    
    with open(conf["conf_path"], 'r') as y:
        args = yaml.load(y, Loader=yaml.FullLoader)
    conf.update(args)
    
    conf["differ_dict_path"] = Path(conf['environs']['thresh_root']) / f'{dataset_mapping[conf["dataset"]]}.json'
    conf["dataset_dir"] = root_data
    conf["differ_types"] = conf['differencer']['types']
    
    return conf