# import some common libraries
import cv2
import os
import json
from tqdm import tqdm


# image_folder_path = '/home/pithreeone/SDC-Repo/2023_final/data/Competition_Image'
image_folder_path = '/home/pithreeone/SDC-Repo/2023_final/data/mini_test/city_7_0/Navtech_Cartesian'
image_files = [f for f in os.listdir(image_folder_path)]
image_files.sort()


gt_path = '/home/pithreeone/SDC-Repo/2023_final/data/mini_test/city_7_0/gt_city_7_0.json'
predict_path = '/home/pithreeone/SDC-Repo/2023_final/vehicle_detection/output_test_3.json'

# Read data from the JSON file
with open(gt_path, 'r') as json_file:
    data_gt = json.load(json_file)

with open(predict_path, 'r') as json_file:
    data_predict = json.load(json_file)

id_gt = 0
id_predict = 0

# for image_file in image_files:
for i in tqdm(range(len(image_files))):
    image_file = image_files[i]
    # Construct the full path to the image file
    image_path = os.path.join(image_folder_path, image_file)

    # Read the image using OpenCV
    img = cv2.imread(image_path)

    
    # ground truth bounding box
    for id_gt in range(id_gt, len(data_gt)):
        sample_token = int(data_gt[id_gt]['sample_token'])
        if sample_token < i+1:
            id_gt += 1
            break
        elif sample_token > i+1:
            break
        
        x_min, y_min = data_gt[id_gt]['points'][1]
        x_max, y_max = data_gt[id_gt]['points'][3]
        id_gt += 1

        # Draw bounding box on the image (for visualization purposes)
        cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)

    # prediction bounding box
    for id_predict in range(id_predict, len(data_gt)):
        sample_token = int(data_predict[id_predict]['sample_token'])
        if sample_token < i+1:
            id_predict += 1
            break
        elif sample_token > i+1:
            break
        
        x_min, y_min = data_predict[id_predict]['points'][1]
        x_max, y_max = data_predict[id_predict]['points'][3]
        id_predict += 1

        # Draw bounding box on the image (for visualization purposes)
        cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), 2)

        
    cv2.imshow("Prediction", img)
    cv2.waitKey(200)
