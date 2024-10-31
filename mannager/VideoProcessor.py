
from typing import Tuple
import time
import cv2
import numpy as np


class VideoProcessor():
    """영상을 프레임단위로 읽어들이거나 내보내기 위한 관리자
    """
    def __init__(self, video_path: str, fps: int, test_method: str, size: int, action_dim=30) :
        """init
        """
        self.method = test_method
        self.video_path = video_path
        self.fps = fps
        self.processed_frames_index = []
        self.num_all, self.num_processed = 0, 0
        self.idx, self.skip_idx = 0, 0
        
        if self.method != 'reducto':
            self.action_dim = action_dim
            
        self.cap = cv2.VideoCapture(self.video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        _, f_init = self.cap.read()
        time.sleep(1.0 / self.fps)
        frame_shape = (1, 3, int(f_init.shape[0]//size), int(f_init.shape[1]//size))
        self.frame_shape = str(frame_shape)
        self.cur_frame, self.last_skip_frame, self.last_processed_frame = f_init, f_init, f_init
    
    
    def read_video(self, skip:bool):
        """현재 프레임(self.cur_frame)을 skip할 것인지 처리할 것인지를 인자로 전달받아, skip/혹은 처리를 진행한 뒤 다음 프레임을 읽어서 반환합니다.
        Args:
            skip (bool): _description_
        Returns:
            _type_: _description_
        """
        if skip:
            self.processed_frames_index.append(self.skip_idx)
            self.last_skip_frame = self.cur_frame
        else:
            self.processed_frames_index.append(self.idx)
            self.last_processed_frame = self.cur_frame
            self.skip_idx = self.idx
            self.num_processed += 1
            
        self.num_all += 1
        
        ret, self.cur_frame = self.cap.read()
        if not ret :
            return False
        
        time.sleep(1.0 / self.fps)
        self.idx += 1
        
        return True
    
    
    def update_segment(self, segment_path):
        self.cap.release()
        self.cap = cv2.VideoCapture(segment_path)
        _, f_init = self.cap.read()
        # time.sleep(1.0 / self.fps)
        self.cur_frame, self.last_skip_frame, self.last_processed_frame = f_init, f_init, f_init
    
    
    def get_frame(self) -> Tuple[np.array, np.array, int]:
        """현재 frame과 이전 frame, 그리고 현재 frame의 idx를 반환한다.

        Returns:
            Tuple[prev_frame, cur_frame, last_skip_frame, idx]:
        """
        return self.cur_frame, self.last_skip_frame, self.last_processed_frame, self.idx
