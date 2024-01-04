
import argparse


import json
import numpy as np
import os
import random
from detectron2.engine import DefaultTrainer, hooks
from utils.trainer import Trainer
from utils.rotated_trainer import RotatedTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader, build_detection_train_loader
from detectron2.structures import BoxMode
from detectron2.evaluation import (
    COCOEvaluator,
    RotatedCOCOEvaluator)
from detectron2.data import build_detection_train_loader
from detectron2.data import DatasetMapper
from detectron2.data import transforms as T
import torch
import cv2

# init params
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", help="Model Name (Ex: faster_rcnn_R_50_FPN_3x)",
                    default='faster_rcnn_R_101_FPN_3x',
                    type=str)

parser.add_argument("--root_folder", help="root folder with radiate dataset",
                default='/home/pithreeone/SDC-Repo/2023_final/data/Bonus_train',
                    type=str)

parser.add_argument("--max_iter", help="Maximum number of iterations",
                    default=1000,
                    type=int)

parser.add_argument("--resume", help="Whether to resume training or not",
                    default=False,
                    type=bool)

parser.add_argument("--dataset_mode", help="dataset mode ('good_weather', 'good_and_bad_weather')",
                    default='good_and_bad_weather',
                    type=str)

# parse arguments
args = parser.parse_args()
model_name = args.model_name
root_dir = args.root_folder
resume = args.resume
dataset_mode = args.dataset_mode
max_iter = args.max_iter

class MyTrainer(DefaultTrainer):

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapper(cfg, is_train=True, augmentations=[
                                                                #   T.RandomFlip(),
                                                                #   T.RandomCrop("absolute", (700, 700)),
                                                                #   T.RandomRotation(angle = [-45, 45]),
                                                                #   T.RandomBrightness(0.7, 1.3),  # Adjust brightness during testing
                                                                #   GaussianBlurTransform(blur_prob=0.3),  # Adjust blur_prob as needed
                                                                  ])
        return build_detection_train_loader(cfg, mapper=mapper)

class GaussianBlurTransform(T.Transform):
    def __init__(self, blur_prob=0.5):
        self.blur_prob = blur_prob

    def apply_image(self, img):
        if np.random.rand() < self.blur_prob:
            img = cv2.GaussianBlur(img, (5, 5), 0)
        return img

    def apply_coords(self, coords):
        return coords

    def inverse(self):
        raise NotImplementedError("GaussianBlurTransform is not invertible")

    def apply_segmentation(self, segmentation):
        return segmentation

def train(model_name, root_dir, dataset_mode, max_iter):

    # output folder to save models
    output_dir = os.path.join('train_results', model_name + '_' + dataset_mode)
    os.makedirs(output_dir, exist_ok=True)

    # get folders depending on dataset_mode
    # folders_train = []
    # folders_test = []
    # for curr_dir in os.listdir(root_dir):
    #     with open(os.path.join(root_dir, curr_dir, 'meta.json')) as f:
    #         meta = json.load(f)
    #     if meta["set"] == "train_good_weather":
    #         folders_train.append(curr_dir)
    #     elif meta["set"] == "train_good_and_bad_weather" and dataset_mode == "good_and_bad_weather":
    #         folders_train.append(curr_dir)
    #     elif meta["set"] == "test":
    #         folders_test.append(curr_dir)

    def gen_boundingbox(bbox, angle):
        theta = np.deg2rad(-angle)
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        points = np.array([[bbox[0], bbox[1]],
                           [bbox[0] + bbox[2], bbox[1]],
                           [bbox[0] + bbox[2], bbox[1] + bbox[3]],
                           [bbox[0], bbox[1] + bbox[3]]]).T

        cx = bbox[0] + bbox[2] / 2
        cy = bbox[1] + bbox[3] / 2
        T = np.array([[cx], [cy]])

        points = points - T
        points = np.matmul(R, points) + T
        points = points.astype(int)

        min_x = np.min(points[0, :])
        min_y = np.min(points[1, :])
        max_x = np.max(points[0, :])
        max_y = np.max(points[1, :])

        return min_x, min_y, max_x, max_y

    def get_radar_dicts():
        dataset_dicts = []
        idd = 0
        # folder_size = len(folders)

        # for folder in folders:
        radar_folder = os.path.join(root_dir, 'Navtech_Cartesian')
        annotation_path = os.path.join(root_dir, 'annotations', 'annotations.json')
        with open(annotation_path, 'r') as f_annotation:
            annotation = json.load(f_annotation)

        radar_files = os.listdir(radar_folder)
        radar_files.sort()

        
        old_image_id = 0

        record = {}
        objs = []
        for object in annotation['annotations']:
            idd += 1

            
            image_id = object['image_id']
            if old_image_id != image_id:
                old_image_id = image_id
                record["annotations"] = objs
                dataset_dicts.append(record)
                record = {}
                objs = []

            # find the image name
            for image in annotation['images']:
                if (int(image['id']) == image_id):
                    file_name = image['file_name']
                    filename = os.path.join(radar_folder, file_name)
                    record["file_name"] = filename
                    record["image_id"] = image_id
                    break
            
            record["height"] = 1600
            record["width"] = 1600
            
            class_id = object['category_id']
            bbox = object['bbox']
            angle = object['angle']
            bbox_detectron2 = gen_boundingbox(bbox, angle)  # You may need to adjust the angle parameter

            # # Add the object to the list

            objs.append({
                "bbox": bbox_detectron2,
                "bbox_mode": BoxMode.XYXY_ABS,  # Assuming bounding box format is XYXY
                "category_id": class_id  # Only one class ('vehicle') is assumed
            })

        record["annotations"] = objs
        dataset_dicts.append(record)

            

        return dataset_dicts



        # for frame_number in range(len(radar_files)):
        #     record = {}
        #     objs = []
        #     bb_created = False
        #     idd += 1
        #     filename = os.path.join(radar_folder, radar_files[frame_number])

        #     if (not os.path.isfile(filename)):
        #         print(filename)
        #         continue
        #     record["file_name"] = filename
        #     record["image_id"] = idd
        #     record["height"] = 1152
        #     record["width"] = 1152

        #     for object in annotation['annotations']:
                


        #         if (object['bboxes'][frame_number]):
        #             class_obj = object['class_name']

        #             if class_obj == 'group_of_pedestrians' or class_obj == 'pedestrian':
        #                 continue
                    
        #             # ## Student implement ###
        #             # TODO
        #             # print(len(annotation))
        #             # bbox = object['bboxes'][frame_number]
        #             # print(object['bboxes'][frame_number])
        #             # # Extract bounding box coordinates and convert them to Detectron2 format
        #             bbox_np = object['bboxes'][frame_number]['position']
        #             bbox_detectron2 = gen_boundingbox(bbox_np, object['bboxes'][frame_number]['rotation'])  # You may need to adjust the angle parameter

        #             # # Add the object to the list
        #             objs.append({
        #                 "bbox": bbox_detectron2,
        #                 "bbox_mode": BoxMode.XYXY_ABS,  # Assuming bounding box format is XYXY
        #                 "category_id": 0  # Only one class ('vehicle') is assumed
        #             })

        #             # # Set bb_created to True to indicate that at least one bounding box is created for this frame
        #             bb_created = True
                    
        #     if bb_created:
        #         record["annotations"] = objs
        #         dataset_dicts.append(record)

        # return dataset_dicts

    dataset_train_name = 'Bonus_train'

    DatasetCatalog.register(dataset_train_name,
                            lambda: get_radar_dicts())
    dicts = get_radar_dicts()
    # print(dicts[-2:])
    MetadataCatalog.get(dataset_train_name).set(thing_classes=['car', 'scooter', 'scooters'])

    # print(len(DatasetCatalog.get(dataset_train_name)))
    # print(MetadataCatalog.get(dataset_train_name).thing_classes)

    cfg_file = os.path.join('test', 'config', model_name + '.yaml')
    
    torch.cuda.empty_cache()
    cfg = get_cfg()
    cfg.OUTPUT_DIR = output_dir
    cfg.merge_from_file(cfg_file)
    cfg.DATASETS.TRAIN = (dataset_train_name,)
    # cfg.DATASETS.TEST = (dataset_test_name,)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 1
    # cfg.SOLVER.STEPS = (7000)
    cfg.SOLVER.MAX_ITER = max_iter
    cfg.SOLVER.BASE_LR = 0.0001
    # cfg.SOLVER.MOMENTUM = 0.5
    cfg.SOLVER.GAMMA = 0.5
    cfg.MODEL.DEVICE = 'cuda'
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 32
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.2
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8, 16, 32, 64, 128]]

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    if cfg.MODEL.PROPOSAL_GENERATOR.NAME == "RRPN":
        trainer = RotatedTrainer(cfg)
    else:
        # trainer = Trainer(cfg)
        trainer = MyTrainer(cfg)

    trainer.resume_or_load(resume=resume)
    trainer.train()


if __name__ == "__main__":
    train(model_name, root_dir, dataset_mode, max_iter)