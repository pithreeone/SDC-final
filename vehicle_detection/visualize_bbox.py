# import some common libraries
import cv2
import os
import json
from tqdm import tqdm


image_folder_path = '/home/pithreeone/SDC-Repo/2023_final/data/Bonus_Image'
# image_folder_path = '/home/pithreeone/SDC-Repo/2023_final/data/mini_test/city_7_0/Navtech_Cartesian'
image_files = [f for f in os.listdir(image_folder_path)]
image_files.sort()


predict_path = '/home/pithreeone/SDC-Repo/2023_final/vehicle_detection/output.json'
# predict2_path = '/home/pithreeone/SDC-Repo/2023_final/vehicle_detection/output_comp_3.json'

# Read data from the JSON file
with open(predict_path, 'r') as json_file:
    data_predict = json.load(json_file)
print(data_predict[0])

# with open(predict2_path, 'r') as json_file:
#     data_predict2 = json.load(json_file)

id_predict = 0
id_predict2 = 0

# Specify the output video file name
output_video_file = 'output_video.mp4'
# Define the codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use other codecs like 'XVID', 'MJPG', etc.
# out = cv2.VideoWriter(output_video_file, fourcc, 12.0, (1152, 1152))


# for image_file in image_files:
for i in tqdm(range(len(image_files))):
    # print(i)
    image_file = image_files[i]
    # Construct the full path to the image file
    image_path = os.path.join(image_folder_path, image_file)

    # Read the image using OpenCV
    img = cv2.imread(image_path)
 
    # prediction bounding box
    # for id_predict2 in range(id_predict2, len(data_predict2)):
    #     sample_token = int(data_predict2[id_predict2]['sample_token'])
        
    #     if sample_token < i+1:
    #         id_predict2 += 1
    #         break
    #     elif sample_token > i+1:
    #         break
        
    #     x_min, y_min = data_predict2[id_predict2]['points'][1]
    #     x_max, y_max = data_predict2[id_predict2]['points'][3]
    #     id_predict2 += 1

    #     # Draw bounding box on the image (for visualization purposes)
    #     cv2.rectangle(img, (int(x_min + 5), int(y_min + 5)), (int(x_max + 5), int(y_max + 5)), (0, 255, 0), 2)

    # prediction bounding box
    for id_predict in range(id_predict, len(data_predict)):
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
        cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255), 2)

    img = cv2.resize(img, (900,900))
    cv2.imshow("Prediction", img)
    # out.write(img)
    cv2.waitKey(50)


# Release the VideoWriter object
# out.release()