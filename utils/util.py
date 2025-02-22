import datetime
import os
import re
import csv
import argparse
from typing import Tuple, Union, Dict


def str2bool(v) :
    if isinstance(v, bool) :
        return v
    if v.lower() in ('true', 'yes', 't') :
        return True
    elif v.lower() in ('false', 'no', 'f') :
        return False
    else :
        raise argparse.ArgumentTypeError('Boolean value expected.')



def save_parameters_to_csv(start_time: str, conf: Dict[str, Union[str, int, bool, float]]):
    end_time = datetime.datetime.now().strftime("%y%m%d-%H%M%S") 
    desired_keys = ['test_method', 'video_path', 'cluster_path', 'model_path', 'idx_list', 'sf', 'rf', 'avg_f1', 'fraction']

    csv_file_path = 'test_config.csv'
    
    existing_data = []
    if os.path.exists(csv_file_path):
        with open(csv_file_path, 'r') as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                existing_data.append(row)

    # Filter and reorder the configuration based on desired keys
    new_row = [start_time] + [end_time] + [str(conf[key]) for key in desired_keys if key in conf]
    existing_data.append(new_row)

    with open(csv_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(existing_data)