"""
represents the environment in which the model interacts; 
it uses replay buffers because we using offline-learning.
"""
import random
from typing import Tuple, List, Union, Dict
from util.cluster import *
from util.obj import *

random.seed(42)


class Environment():
    """강화학습 agent와 상호작용하는 environment
    """
    def __init__(self, conf:Dict[str, Union[str, bool, int, float]], communicator, video_processor, run:bool=False):
        """init

        Args:
            conf (Dict[str, Union[bool, int, float]]): train/test setting
            run (bool, optional): testing을 수행하는가?
        
        Calls:
            utils.get_state.cluster_init(): cluster 초기화
            utils.get_state.cluster_load(): cluster_path에서 불러옴
            self.reset(): Env reset
        """
        self.run = run
        self.fps = conf['fps']
        self.action_dim = self.fps
        self.debug_mode = conf['debug_mode']
        self.communicator = communicator
        self.video_processor = video_processor
        
        if not run:
            self.target_f1 = conf['target_f1']
            self.threshold = 1 - self.target_f1
            self.psi_1 = conf['psi_1']
            self.psi_2 = conf['psi_2']        
            
        self.model = load_cluster(conf['cluster_path'])
        
        self.reset()

    
    def reset(self, show_log:bool=False) -> Union[int, List[float]]:
        """Env reset: Arrival data, reward, idx, frame_reader, state 초기화 

        Args:
            show_log (bool, optional): command line에 이번 에피소드 전체의 transition을 출력할 것인가?

        Returns:
            Union[int, List[float, float]]: initial state를 반환한다.
            
        Calls:
            utils.cal_quality.get_FFT(): run 모드일 때 FFT 계산을 위해 사용한다.
            utils.get_state.cluster_pred: train, Q-learning에서 state를 구하기 위해 사용한다.
        """
        self.video_processor.reset()
        self.sum_reward, self.sum_send_frame = 0, 0
        self.show_log = show_log
        self.prev_frame, self.cur_frame, self.idx = self.video_processor.get_frame()

        self.state = self.__observe_environment(self.prev_frame, self.cur_frame)
            
        self.trans_list = []
        if self.show_log :   
            self.logList = []
        
        self.reward_print_count = 0
        
        return self.state


    def __observe_environment(self, prev_frame, cur_frame):
        """주어진 frame들을 이용하여 cluster 결과로 얻어진 state를 반환한다.

        Args:
            prev_frame (List): 
            cur_frame (List): 
        """
        prev_chunk_list = get_chunk(prev_frame)
        cur_chunk_list = get_chunk(cur_frame)
        diff_vector = []
        
        for c1, c2 in zip(prev_chunk_list, cur_chunk_list):
            diff_vector.append(get_diff(c1, c2))
        
        state = get_state(self.model, diff_vector)
        
        return state


    def step(self, action:int) :
        """인자로 전달된 action을 수행하고, 행동에 대한 reward를 계산하여 결과적으로 얻은 trans를 반환합니다.

        Args:
            action (int): 수행할 action [0, fps]

        Returns:
            Tuple[next_state, reward, done]
        
        Calls:
            __triggerL Lyapunov based guide가 새로 들어왔을 때 변수 갱신, 리워드 계산을 위해 호출
        """
        #TODO: communicator
        done = False
        ret = self.video_processor.read_video(skip=action)
        if not ret:
            if self.run:
                idx_list = self.video_processor.processed_frames_index
                return idx_list, True
            return self.trans_list, True

        self.__trigger(action)
        
        return self.trans_list, done


    def __trigger(self, action:int) -> float:
        """action을 취했을 때, 이를 적용하고 reward를 계산하여 trans_list를 만듭니다.
        Args:
            action (int): 수행하고자 했던 action

        Returns:
            float: reward            
        """
        reward = 0
        self.sum_send_frame += (self.action_dim - action)
        
        self.trans_list = []
        self.trans_list.append(self.state)
        self.trans_list.append(action)
        
        self.prev_frame, self.cur_frame, self.idx = self.video_processor.get_frame()
        self.state = self.__observe_environment(self.prev_frame, self.cur_frame)
        self.trans_list.append(self.state)
        
        if not self.run :
            reward = self.__reward_function()
        
        self.trans_list.append(reward)
        
        return

    # not used
    def __cal_error(self, last_idx, process_idx):
        error = 1 - get_F1_with_idx(last_idx, process_idx, self.video_processor.video_path)
        
        return error
    
    # not used
    def __reward_function(self):
        """reward를 계산합니다.
        """
        s, a, s_prime = self.trans_list
        
        last_idx = self.idx - a - 1
        skip_idx = self.idx - 1
        
        error = self.__cal_error(last_idx, skip_idx)
        if error <= self.threshold :
            r = self.psi_1 * (a+1)
        else :
            r = -1 * self.psi_2 * a

        if self.debug_mode and self.reward_print_count < 10:
            self.reward_print_count += 1
            print("reward:", r)
            print("error status:", error)
            print()
        
        self.sum_reward += r
        
        if self.show_log :
            self.logList.append("s(t): {:2d}\tu(t): {:2d}\ts(t+1): {:2d}\tr(t): {:.5f}".format(s[0], a, s_prime[0], r))
            
        return r


    def show_trans(self) :
        """show_log가 true일 때, 에피소드 전체에서 각각의 transition을 모두 출력합니다.
        """
        for row in self.logList :
            print(row)
        return