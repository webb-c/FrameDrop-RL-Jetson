import datetime
from src.cao.utils.util import get_optimal_parameter
from typing import Tuple, Union, Dict
from src.cao.test import test_model

def testor_cao(conf:Dict[str, Union[str, int, bool, float]], communicator, video_processor) -> bool:
    
    print("Ready ...")
    start_time = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    
    sf, rf, accuracy = get_optimal_parameter(conf)
    conf['sf'] = sf
    conf['rf'] = rf
    conf['avg_f1'] = accuracy
    
    fraction_value = test_model(conf, start_time, video_processor, communicator, conf['jetson_mode'])
    rounded_fraction = round(fraction_value, 4)
    conf['fraction'] = rounded_fraction
    
    if conf['jetson_mode']:
        communicator.get_message()
        communicator.send_message("finish")
        communicator.close_queue()
        
    finish_time = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    
    return True, finish_time


