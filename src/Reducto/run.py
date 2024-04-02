import os
import cv2
import argparse
import functools
import multiprocessing as mp
from pathlib import Path

from tqdm import tqdm
import mongoengine
import yaml
from pyprnt import prnt

from simulator import *
from util.data_loader import dump_json
from util.differencer import DiffComposer
from util.model import Segment, Inference, InferenceResult, DiffVector, FrameEvaluation
from util.utils import generate_thresholds

import tensorflow as tf
import tf_slim as slim

tf.compat.v1.disable_eager_execution()

ROOT = 'D:/VSC/INFOCOM/baselines/reducto/'
conf_dir = ROOT + 'pipelines/'
hash_table_dir = ROOT + 'hashmap/'
log_dir = ROOT + 'logs/'
segment_pattern = 'segment???.mp4'
test_root = 'D:/VSC/INFOCOM/DATASET/test/split/'
train_root = 'D:/VSC/INFOCOM/DATASET/train/split/'

dataset_mapping = {
    'JK-1' : 'JK',
    'JK-2' : 'JK',
    'SD-1' : 'SD',
    'SD-2' : 'SD',
    'JN-1' : 'JN'
}

datasets = [
    {
        'dataset': 'JK',
        'subsets': ['subset1', 'subset2'],
        'reducto_queries_90': [{'metric': 'mAP-all', 'differ': 'edge', 'distance': 1.0, 'safe': 0.0025, 'target_acc': 0.90}],
        'reducto_queries_70': [{'metric': 'mAP-all', 'differ': 'edge', 'distance': 5.0, 'safe': 0.05, 'target_acc': 0.70}],
        'properties': {'fps': 30},
    },
    {
        'dataset': 'SD',
        'subsets': ['subset1', 'subset2'],
        'reducto_queries_90': [{'metric': 'mAP-all', 'differ': 'edge', 'distance': 1.0, 'safe': 0.0025, 'target_acc': 0.90}],
        'reducto_queries_70': [{'metric': 'mAP-all', 'differ': 'edge', 'distance': 5.0, 'safe': 0.05, 'target_acc': 0.70}],
        'properties': {'fps': 30},
    },
    {
        'dataset': 'JN',
        'subsets': ['subset1'],
        'reducto_queries_90': [{'metric': 'mAP-all', 'differ': 'edge', 'distance': 1.0, 'safe': 0.0025, 'target_acc': 0.90}],
        'reducto_queries_70': [{'metric': 'mAP-all', 'differ': 'edge', 'distance': 5.0, 'safe': 0.05, 'target_acc': 0.70}],
        'properties': {'fps': 30},
    },
]



def str2bool(v) :
    if isinstance(v, bool) :
        return v
    if v.lower() in ('true', 'yes', 't') :
        return True
    elif v.lower() in ('false', 'no', 'f') :
        return False
    else :
        raise argparse.ArgumentTypeError('Boolean value expected.')


def run_setting(args):
    args["conf_path"] = conf_dir + args["dataset"] + ".yaml"
        
    with open(args["conf_path"], 'r') as y:
        conf = yaml.load(y, Loader=yaml.FullLoader)
    conf.update(args)
    
    check_dataset(conf["dataset"], conf["mode"])
    
    if conf['mode'] == "test":
        conf["differ_dict_path"] = Path(conf['environs']['thresh_root']) / f'{dataset_mapping[args["dataset"]]}.json'
        conf["dataset_dir"] = test_root
    else:
        conf["differ_dict_path"] = Path(conf['environs']['thresh_root']) / f'{args["dataset"]}.json'
        conf["dataset_dir"] = train_root
    
    conf["differ_types"] = conf['differencer']['types']
    

    if conf['mode'] == "test":
        conf["hash_table_path"] = hash_table_dir + dataset_mapping[conf["dataset"]] + "_" + str(conf["safe_zone"]) + "_" + str(conf["target_acc"]) + ".pkl"
    else:
        conf["hash_table_path"] = hash_table_dir + conf["dataset"]  + "_" + str(conf["safe_zone"]) + "_" + str(conf["target_acc"]) + ".pkl"
    conf["log_path"] = log_dir + conf['dataset'] + "_" + str(conf['distance']) + "_" + str(conf["safe_zone"]) + "_" + str(conf["target_acc"])
    
    return conf


def check_dataset(dataset_name, mode):
    if mode == "test":
        root = test_root
    else:
        root = train_root
        
    if os.path.isdir(os.path.join(root, dataset_name)):
        print(f"dataset {dataset_name} is valid.")
    else:
        print(f"dataset {dataset_name} is invalid, Start making segment.")
        origin_path = 'D:/VSC/INFOCOM/DATASET/' + mode
        input_path = Path(origin_path) / dataset_name / ".mp4"
        output_folder = os.path.join(root, dataset_name)
        make_segment_data(input_path, output_folder)



def make_segment_data(input_path, output_folder, seg_len=4):
    frames_per_segment = fps * seg_len
    video_capture = cv2.VideoCapture(input_path)
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    os.makedirs(output_folder, exist_ok=True)
    frame_count = 0
    segment_count = 1

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        if frame_count == 0:
            segment_output_path = os.path.join(output_folder, f"segment{str(segment_count).zfill(3)}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(segment_output_path, fourcc, fps, (width, height))

        video_writer.write(frame)
        frame_count += 1

        if frame_count >= frames_per_segment:
            video_writer.release()
            print(f"Segment {segment_count} created.")
            segment_count += 1
            frame_count = 0
            
    video_writer.release()
    print(f"Segment {segment_count} created.")
    video_capture.release()


def get_segments(conf):
    segments = []
    p = Path(conf["dataset_dir"]) / conf["dataset"] / conf["subset"]
    segments += [f for f in sorted(p.iterdir()) if f.match(segment_pattern)]
    
    return segments


def get_instances(conf):
    no_session = False

    model, evaluator = None, None
    differ = DiffComposer.from_jsonfile(conf['differ_dict_path'], conf['differ_types'])
    
    return model, differ, evaluator


def testor_reducto(args):
    """main method for testing Reducto
    Args:
        args (dict): testing argument dictionary
    """
    conf = run_setting(args)
    prnt(conf)
    
    # mongoengine.connect(conf["dataset"], host=conf['mongo']['host'], port=conf['mongo']['port'])
    # print(f'connected to {conf["mongo"]["host"]}:{conf["mongo"]["port"]} on dataset {conf["dataset"]}')
    
    segments = get_segments(conf)
    model, differ, evaluator = get_instances(conf)
    
    simulator = Reducto(datasets)
    
    diff_results, fraction_change = simulator.run(conf, conf['dataset'], ['subset0'])
    avg_fraction = sum(fraction_change) / len(fraction_change)
    
    return True


if __name__ == "__main__":
    opt = parse_args()
    args = dict(**opt.__dict__)
    ret = main(args)
    
    if not ret:
        print("Testing ended abnormally.")
