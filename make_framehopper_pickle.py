import os
import cv2
from typing import Tuple, List, Dict, Union
import numpy as np

from src.FrameHopper.util.cluster import *
from src.FrameHopper.util.obj import get_chunk, get_diff

def path_manager(video_name: str) -> str :
    """ video path를 전달받아서 cluster, detection 경로를 지정하여 반환합니다.

    Args:
        video_path (str): 학습에 사용할 영상의 경로
    Returns:
        Tuple[cluster_path, detection_path,
    """
    root_cluster  = "./model/FrameHopper/cluster/new/"
    root_video = "./data/train/"
    cluster_path = os.path.join(root_cluster, video_name + ".pkl")
    video_path  = os.path.join(root_video, video_name + ".mp4")
    
    return cluster_path, video_path


def verifier(video_path, cluster_path: str):
    """경로를 전달받아서 데이터가 존재하는지 검증하고 없다면 만들어냅니다.

    Args:
        conf (Dict[str, Union[bool, int, float]]): train argument
        cluster_path (str): _description_
        detection_path (str): _description_
    """
    if not os.path.exists(cluster_path):
        print("start making cluster model ...")

        cluster = init_cluster(state_num=10)
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        frame_list = []
        train_data = []
        while True :
            ret, frame = cap.read()
            if not ret :
                cap.release()
                cluster = train_cluster(cluster, np.array(train_data), cluster_path)
                break
            
            frame_count += 1
            frame_list.append(frame)
            if len(frame_list) > 30:    
                for k in range(1, 31):
                    diff_vector = []
                    prev_chunk_list = get_chunk(frame_list[0])
                    cur_chunk_list = get_chunk(frame_list[k])
                    for c1, c2 in zip(prev_chunk_list, cur_chunk_list):
                        diff_vector.append(get_diff(c1, c2))
                    train_data.append(diff_vector)
                frame_list.pop(0)
    
            if frame_count == 90:
                #print(np.array(train_data).shape)
                cluster = train_cluster(cluster, np.array(train_data), cluster_path)
                train_data = []
                frame_count = 0
    
        save_cluster(cluster, cluster_path)
        print("finish making cluster model in", cluster_path)
        print()


def main(video_name: str) -> bool :
    cluster_path, video_path = path_manager(video_name)
    verifier(video_path, cluster_path)
    
    return True


if __name__ == "__main__":
    for video_name in ["JK", "SD", "JN"]:
        ret = main(video_name)