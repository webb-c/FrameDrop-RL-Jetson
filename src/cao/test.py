import torch
import time, csv, os, re
import numpy as np
import cv2
import shutil
import torch.nn.functional as F
from src.cao.utils.image_util import resize, downscaling_for_send_with_dummy, downscaling
from src.cao.utils.coarse_segmentation import CoarseSegmentationModel
import os

def test_model(conf, start_time, video_processor, communicator=None, jetson_mode=False):
    device = conf['device']
    
    model = CoarseSegmentationModel().to(device)
    model.load_state_dict(torch.load(conf['weight_path'], weights_only=True))
    model.eval()
    
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
        send_frame_shape = downscaling_for_send_with_dummy(input_frame, output, conf['sf'], conf['rf']) 
        # _ = downscaling(input_frame, output, conf['sf'], conf['rf']) 
        # send_frame_shape = (1, 1, 1, 600)

    fraction_value = np.prod(send_frame_shape) / (3*conf['img_size']*conf['img_size'])
    return fraction_value