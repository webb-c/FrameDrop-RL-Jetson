import torch
import time, csv, os, re
import numpy as np
import cv2
import shutil
import torch.nn.functional as F
from utils.util import path_maker
from utils.image_util import resize, downscaling_for_send, change_frame_shape
from utils.coarse_segmentation import CoarseSegmentationModel
import os

def test_model(conf, start_time, video_processor, communicator=None, jetson_mode=False):
    error = False
    device = conf['device']
    
    model = CoarseSegmentationModel().to(device)
    model.load_state_dict(torch.load(conf['weight_path'], weights_only=True))
    model.eval()
    root_source = 'datasets/source'
    
    total_frames = video_processor.total_frames
    step = 0
    while True:
        print(f"Update progress: [{step}/{total_frames}]")
        ret = video_processor.read_video(skip=False)
        step += 1
        if not ret:
            break
        frame, _, _, _ = video_processor.get_frame()
        input_frame = torch.from_numpy(frame).permute(2, 0, 1)
        input_frame = input_frame.unsqueeze(0)
        input_frame = resize(input_frame, input_frame.shape[-1], (conf['img_size'], conf['img_size'])).to(device).float()

        output = model(input_frame)
        origin_pathes, compressed_pathes = downscaling_for_send(input_frame, output, conf['sf'], conf['rf']) 
        send_frame_shape = change_frame_shape(origin_pathes, compressed_pathes)

        #!TODO: communicator
        if jetson_mode:
            communicator.get_message()
            communicator.send_message("action")
            communicator.get_message()
            communicator.send_message(send_frame_shape)

    
    return True