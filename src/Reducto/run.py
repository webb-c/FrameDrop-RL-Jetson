import datetime
from src.Reducto.simulator import Reducto


datasets = [
    {
        'dataset': 'JK',
        'subsets': ['subset1', 'subset2'],
        'reducto_queries_90': [{'metric': 'mAP-all', 'differ': 'edge', 'distance': 2.0, 'safe': -0.0025, 'target_acc': 0.70}],
        'reducto_queries_70': [{'metric': 'mAP-all', 'differ': 'edge', 'distance': 1.0, 'safe': 0.025, 'target_acc': 0.90}],
        'properties': {'fps': 30},
    },
    {
        'dataset': 'SD',
        'subsets': ['subset1', 'subset2'],
        'reducto_queries_90': [{'metric': 'mAP-all', 'differ': 'edge', 'distance': 3.0, 'safe': -0.05, 'target_acc': 0.70}],
        'reducto_queries_70': [{'metric': 'mAP-all', 'differ': 'edge', 'distance': 2.0, 'safe': -0.05, 'target_acc': 0.90}],
        'properties': {'fps': 30},
    },
    {
        'dataset': 'JN',
        'subsets': ['subset1'],
        'reducto_queries_90': [{'metric': 'mAP-all', 'differ': 'edge', 'distance': 1.0, 'safe': 0.0025, 'target_acc': 0.70}],
        'reducto_queries_70': [{'metric': 'mAP-all', 'differ': 'edge', 'distance': 0.25, 'safe': 0.0025, 'target_acc': 0.90}],
        'properties': {'fps': 30},
    },
]


def testor_reducto(conf, communicator, video_processor):
    """main method for testing Reducto
    Args:
        args (dict): testing argument dictionary
    """
    simulator = Reducto(datasets, communicator, video_processor, conf['jetson_mode'], conf['debug_mode'])
    
    diff_results, fraction_change = simulator.run(conf, conf['dataset'], ['subset0'])
    idx_list = [result['selected_frames'] for result in diff_results]
    avg_fraction = sum(fraction_change) / len(fraction_change)
    
    if conf['jetson_mode']:
        simulator.communicator.get_message()
        simulator.communicator.send_message("finish")
        simulator.communicator.close_queue()
        
    finish_time = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    
    conf['idx_list'] = idx_list
    conf['fraction'] = avg_fraction
    
    return True, finish_time

