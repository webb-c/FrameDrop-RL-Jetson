
from typing import Tuple
import cv2
import numpy as np


class VideoProcessor():
    """영상을 프레임단위로 읽어들이거나 내보내기 위한 관리자
    """
    def __init__(self, video_path: str, fps: int, test_method: str, action_dim=30) :
        """init
        """
        self.method = test_method
        self.video_path = video_path
        self.fps = fps
        self.processed_frames_index = []
        self.num_all, self.num_processed = 0, 0

        if self.method != 'reducto':
            self.action_dim = action_dim
            self.cap = cv2.VideoCapture(self.video_path)
            _, f_init = self.cap.read()
            self.prev_frame, self.last_skip_frame, self.cur_frame = f_init, f_init, f_init
            self.idx = 0
            self.num_all, self.num_processed = 0, 0
            self.frame_shape = str(f_init.shape[:2])
        
        if self.method == 'reducto':
            #TODO: reducto VideoProcessor Design
            segment_list = []
    
    
    def read_video(self, skip:int) -> bool:
        """현재 프레임을 저장하고, 주어진 skip길이만큼 skip한다.
        Args:
            skip (int): skip하는 프레임의 개수 [0, fps]

        Returns:
            bool: 영상이 끝났는가?
        """
        # 현재 프레임은 처리
        if self.method == 'frameHopper':
            self.processed_frames_index.append(self.idx)
            self.num_all += 1
            self.num_processed += 1
            
            # 마지막으로 처리한 프레임 갱신
            self.prev_frame = self.cur_frame
            skip_idx = self.idx
            
            # skip
            for _ in range(skip):
                ret, _ = self.cap.read()
                if not ret :
                    return False
                self.processed_frames_index.append(skip_idx)
                self.idx += 1
                self.num_all += 1

            # 다음 프레임 read
            ret, self.cur_frame = self.cap.read()
            if not ret :
                return False
            self.idx += 1
                
            return True
        
        elif self.method == 'lrlo':
            skip_idx = self.idx - 1
            for _ in range(skip):
                self.processed_frames_index.append(skip_idx)
                self.prev_frame = self.cur_frame
                
                ret, self.cur_frame = self.cap.read()
                if not ret :
                    return False
                self.idx += 1
                self.num_all += 1

            self.last_skip_frame = self.prev_frame
            for _ in range(self.action_dim - skip):
                self.processed_frames_index.append(self.idx)
                self.prev_frame = self.cur_frame
                
                ret, self.cur_frame = self.cap.read()
                if not ret : 
                    return False
                self.idx += 1
                self.num_all += 1
                self.num_processed += 1
                
            return True

        elif self.method == 'reducto':
            #TODO: reducto video read with VideoProcessor
            
            return True
        
    
    def get_frame(self) -> Tuple[np.array, np.array, int]:
        """현재 frame과 이전 frame, 그리고 현재 frame의 idx를 반환한다.

        Returns:
            Tuple[prev_frame, cur_frame, last_skip_frame, idx]:
        """
        return self.prev_frame, self.cur_frame, self.last_skip_frame, self.idx
