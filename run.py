from pyprnt import prnt
import datetime
from typing import Union, Dict
from mannager.Parser import parse_args, parse_reducto_test, parse_frameHopper_test, parse_lrlo_test
from mannager.VideoProcessor import VideoProcessor
from mannager.Communicator import Communicator
from src.FrameHopper.run import testor_frameHopper
from src.LRLO.run import testor_lrlo
from src.Reducto.run import testor_reducto
from utils.util import save_parameters_to_csv


def main(conf:Dict[str, Union[str, int, bool, float]]) -> bool:
    
    if conf['test_method'] == "Reducto":
        conf = parse_reducto_test(conf)
        testor = testor_reducto
    elif conf['test_method'] == "FrameHopper":
        conf = parse_frameHopper_test(conf)
        testor = testor_frameHopper
    elif conf['test_method'] == "LRLO":
        conf = parse_lrlo_test(conf)
        testor = testor_lrlo
    
    communicator = Communicator(queue_name=conf['test_method'], buffer_size=200000, debug_mode=conf['debug_mode'])
    video_processor = VideoProcessor(video_path=conf['video_path'], fps=conf['fps'], test_method=conf['test_method'])
    
    start_time = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    prnt(conf)
    
    ret, finish_time = testor(conf=conf, communicator=communicator, video_processor=video_processor)
    
    print("Testing Finish !")
    print("\n✱ start time :\t", start_time)
    print("✱ finish time :\t", finish_time)
    
    save_parameters_to_csv(conf)
    
    return True


if __name__ == "__main__":
    opt = parse_args()
    conf = dict(**opt.__dict__)
    ret = main(conf)
    
    if not ret:
        print("Testing ended abnormally.")
