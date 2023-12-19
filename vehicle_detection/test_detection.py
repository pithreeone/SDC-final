import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import os

# radiate sdk
import sys
sys.path.insert(0, '..')
import radiate

# path to the sequence
root_path = '/data/your data path'
sequence_name = 'city_7_0' # just for example

network = 'faster_rcnn_R_50_FPN_3x' # just for example
setting = 'good_and_bad_weather_radar' # just for example

# time (s) to retrieve next frame
dt = 0.25

# load sequence
seq = radiate.Sequence(os.path.join(root_path, sequence_name), config_file='../config/config.yaml')

cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(os.path.join('test','config' , network + '.yaml'))
cfg.MODEL.DEVICE = 'cuda'
cfg.MODEL.WEIGHTS = 'Please use absolute path /xxxx/xxxxx.pth'
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (vehicle)
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.2
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8, 16, 32, 64, 128]]
predictor = DefaultPredictor(cfg)

for t in np.arange(seq.init_timestamp, seq.end_timestamp, dt):
    output = seq.get_from_timestamp(t)
    if output != {}:
        radar = output['sensors']['radar_cartesian']
        camera = output['sensors']['camera_right_rect']
        predictions = predictor(radar)

        predictions = predictions["instances"].to("cpu")
        boxes = predictions.pred_boxes 

        objects = []

        for box in boxes:


            ### Student implement ###
            # TODO


            
            objects.append({'bbox': {'position': ??, 'rotation': ??}, 'class_name': 'vehicle'})

        radar = seq.vis(radar, objects, color=(255,0,0))

        bboxes_cam = seq.project_bboxes_to_camera(objects,
                                                seq.calib.right_cam_mat,
                                                seq.calib.RadarToRight)
        # camera = seq.vis_3d_bbox_cam(camera, bboxes_cam)
        camera = seq.vis_bbox_cam(camera, bboxes_cam)

        cv2.imshow('radar', radar)
        cv2.imshow('camera_right_rect', camera)
        # You can also add other sensors to visualize
        cv2.waitKey(1)
