import datetime
from typing import Tuple, Union, Dict
from agent import Agent
from environment import Environment


def testor_lrlo(conf:Dict[str, Union[str, int, bool, float]], communicator, video_processor) -> bool:
    env = Environment(conf, communicator, video_processor, run=True)
    agent = Agent(conf, run=True)
    done = False
    s = env.reset()
    a_list = []
    
    step = 0
    print("Ready ...")
    
    while not done:
        require_skip = conf['action_dim'] - env.target_A
        if conf["debug_mode"]:
            print("require_skip: ", require_skip)
        a = agent.get_action(s, require_skip, False)
        a_list.append(a)
        s, _, done = env.step(a)
        if done:
            break
        step += 1

    if conf['jetson_mode']:
        env.communicator.get_message()
        env.communicator.send_message("finish")
        env.communicator.close_queue()
        
    finish_time = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    
    fraction_value = env.video_processor.num_processed / env.video_processor.num_all 
    rounded_fraction = round(fraction_value, 4)
    
    conf['idx_list'] = s
    conf['fraction'] = rounded_fraction
    
    return True, finish_time
