import multiprocessing as mp
from pathlib import Path
from pyprnt import prnt
from simulator import *
from util.differencer import DiffComposer

import tensorflow as tf
import tf_slim as slim


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


def get_segments(conf):
    segments = []
    p = Path(conf["dataset_dir"]) / conf["dataset"] / conf["subset"]
    segments += [f for f in sorted(p.iterdir()) if f.match(segment_pattern)]
    
    return segments


def testor_frameHopper(conf:Dict[str, Union[str, int, bool, float]], communicator, video_processor) -> bool:
    env = Environment(conf, communicator, video_processor, run=True)
    agent = Agent(conf, run=True)
    done = False
    s = env.reset()
    a_list = []
    
    step = 0
    print("Ready ...")
    
    while not done:
        a = agent.get_action(s, False)
        a_list.append(a)
        trans, done = env.step(a)
        if done:
            break
        _, _, s, _ = trans
        step += 1
    
    if conf['jetson_mode']:
        env.communicator.get_message()
        env.communicator.send_message("finish")
        env.communicator.close_queue()
        
    finish_time = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    
    fraction_value = env.video_processor.num_processed / env.video_processor.num_all 
    rounded_fraction = round(fraction_value, 4)
    
    conf['idx_list'] = trans
    conf['fraction'] = rounded_fraction
    
    return True, finish_time



def testor_reducto(conf):
    """main method for testing Reducto
    Args:
        args (dict): testing argument dictionary
    """
    segment_list = get_segments(conf)
    differ = DiffComposer.from_jsonfile(conf['differ_dict_path'], conf['differ_types'])
    
    simulator = Reducto(datasets)
    
    diff_results, fraction_change = simulator.run(conf, conf['dataset'], ['subset0'])
    avg_fraction = sum(fraction_change) / len(fraction_change)
    
    return True

