import json
from math import *

file_path = '/home/pithreeone/SDC-Repo/2023_final/vehicle_detection/output_comp_3.json'
file_new_path = '/home/pithreeone/SDC-Repo/2023_final/vehicle_detection/output_comp_3_filter.json'
with open(file_path, 'r') as json_file:
    data_raw = json.load(json_file)

# pre-processing
# Data-structure of data_temp 
# [
#     [None] (token0)
#     [      (token1)
#         [[x1, y1], [x2, y2], [x3, y3], [x4, y4], score]
#     ]
#     [None] (token2)
#     [None] (token3)
#     [      (token4)
#         [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
#         [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
#         [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
#     ]
# ]


image_num = int(data_raw[-1]['sample_token'])
data_filtered = []
data_temp = []
for _ in range(image_num + 1):
    data_temp.append([])

for i in range(len(data_raw)):
    id = int(data_raw[i]['sample_token'])
    temp = data_raw[i]['points']
    temp.append(data_raw[i]['score'])
    data_temp[id].append(temp)

# print(data_temp[-1])

# processing
object = []
# image_num = 100
dist_thresh = 30 # unit: pixel
for i in range(1, image_num):
    # print("-----------%d", i)
    # Add the tracking object
    if i == 1:
        # print(data_temp[i])
        data_temp[i][0].append(i)
        # print(data_temp[i])
        object = (data_temp[i])
        continue

    # print(object)
    for j in range(len(data_temp[i])):
        # print(data_temp[i][j])
        x_min, y_min = data_temp[i][j][1]
        x_max, y_max = data_temp[i][j][3]
        x_c = (x_min + x_max) / 2
        y_c = (y_min + y_max) / 2
        # print(x_c, y_c)
        d_min = 1e8
        id = -1

        # find the nearest object previously
        for k in range(len(object)):
            # print(object[k])
            object_x = (object[k][1][0] + object[k][3][0]) / 2
            object_y = (object[k][1][1] + object[k][3][1]) / 2
            d = sqrt(pow((object_x - x_c), 2) + pow((object_y - y_c), 2))
            if d < d_min:
                d_min = d
                id = k
        # print(dist, k)
                
        # object = []
        if d_min < dist_thresh:
            temp_dict = {}
            temp_dict['sample_token'] = str(i).zfill(6)
            temp_dict['name'] = 'car'
            points = data_temp[i][j][:4]
            temp_dict['points'] = points
            temp_dict['score'] = data_temp[i][j][4]
            data_filtered.append(temp_dict)
            object.append(data_temp[i][j])
            # print(object)

    for j in range(len(data_temp[i])):
        data_temp[i][j].append(i)
        object.append(data_temp[i][j])
    
    
    object[:] = [item for item in object if item[5] > i - 1]

with open(file_new_path, 'w') as json_file:
    json.dump(data_filtered, json_file, indent=2)  # indent for pretty formatting (optional)  
        