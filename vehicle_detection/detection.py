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
import json
from tqdm import tqdm

# path to the sequence
sequence_name = 'city_7_0' # just for example

network = 'faster_rcnn_R_50_FPN_3x' # just for example
setting = 'good_and_bad_weather_radar' # just for example

# time (s) to retrieve next frame
dt = 0.25

cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(os.path.join('test','config' , network + '.yaml'))
cfg.MODEL.DEVICE = 'cpu'
cfg.MODEL.WEIGHTS = '/home/pithreeone/SDC-Repo/2023_final/vehicle_detection/train_results/faster_rcnn_R_50_FPN_3x_good_and_bad_weather/model_final.pth'
# cfg.MODEL.WEIGHTS = '/home/pithreeone/SDC-Repo/2023_final/vehicle_detection/weights/faster_rcnn_R_50_FPN_3x_good_and_bad_weather_radar.pth'
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (vehicle)
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.2
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.65
cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8, 16, 32, 64, 128]]
predictor = DefaultPredictor(cfg)

# image_folder_path = '/home/pithreeone/SDC-Repo/2023_final/data/Competition_Image'
image_folder_path = '/home/pithreeone/SDC-Repo/2023_final/data/mini_test/city_7_0/Navtech_Cartesian'
image_files = [f for f in os.listdir(image_folder_path)]
image_files.sort()
# print(image_files)

file_path = "output.json"
data = []

# for image_file in image_files:
for i in tqdm(range(len(image_files))):
    image_file = image_files[i]
    # Construct the full path to the image file
    image_path = os.path.join(image_folder_path, image_file)

    # Read the image using OpenCV
    img = cv2.imread(image_path)

    # Use your model to make predictions
    outputs = predictor(img)

    # Access the instances from the output
    instances = outputs["instances"]

    # Get the bounding boxes
    boxes = instances.pred_boxes.tensor.numpy()  # numpy array of bounding boxes
    scores = instances.scores.numpy()

    temp = {}
    temp["sample_token"] = image_file[:-4]
    temp["name"] = "car"

    # Loop through each bounding box
    for box, score in zip(boxes, scores):
        x_min, y_min, x_max, y_max = box
        x_min = float(x_min)
        y_min = float(y_min)
        x_max = float(x_max)
        y_max = float(y_max)

        # Process the bounding box as needed
        temp["points"] = [[x_min, y_max], [x_min, y_min], [x_max, y_min], [x_max, y_max]]
        temp["score"] = float(score)

        # Draw bounding box on the image (for visualization purposes)
        cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
        data.append(temp)

        with open(file_path, 'w') as json_file:
            json.dump(data, json_file, indent=2)  # indent for pretty formatting (optional)  

    # Process the outputs or visualize the results as needed
    v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=0.7)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow("Prediction", out.get_image()[:, :, ::-1])
    cv2.waitKey(1)
    # print(image_file)
    # cv2.destroyAllWindows()