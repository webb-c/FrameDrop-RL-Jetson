import datetime
from utils.util import get_optimal_parameter, cal_fraction
from typing import Tuple, Union, Dict
from test import test_model

def testor_cao(conf:Dict[str, Union[str, int, bool, float]], communicator, video_processor) -> bool:
    
    print("Ready ...")
    start_time = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    
    #!TODO
    sf, rf, accuracy = get_optimal_parameter(conf['latency_constraint'])
    fraction_value = cal_fraction(sf, rf)
    rounded_fraction = round(fraction_value, 4)
    
    conf['accuracy'] = accuracy
    conf['fraction'] = rounded_fraction
    
    ret = test_model(conf, start_time, video_processor, communicator, jetson_mode=True)
    
    if conf['jetson_mode']:
        communicator.get_message()
        communicator.send_message("finish")
        communicator.close_queue()
        
    finish_time = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    
    return True, finish_time


