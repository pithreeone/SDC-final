import json
import matplotlib.pyplot as plt

# Specify the path to your JSON file
json_file_path = '/home/pithreeone/SDC-Repo/2023_final/vehicle_detection/train_results/faster_rcnn_R_101_FPN_3x_good_and_bad_weather/metrics_6.json'

# Read the JSON file
with open(json_file_path, 'r') as file:
    # Load the JSON data
    data = file.read()

# Split the data into individual JSON objects
json_objects = data.strip().split('\n')

# Parse each JSON object
parsed_data = [json.loads(obj) for obj in json_objects]

# Assuming parsed_data is your list of dictionaries from the JSON
# Record information: "data_time", "eta_seconds", "fast_rcnn/cls_accuracy", "fast_rcnn/false_negative", "fast_rcnn/fg_cls_accuracy", 
# "iteration", "loss_box_reg", "loss_cls", "loss_rpn_cls", "loss_rpn_loc", "lr", "roi_head/num_bg_samples", "roi_head/num_fg_samples",
# "rpn/num_neg_anchors", "rpn/num_pos_anchors", "time", "total_loss"}

x_label = 'iteration'
y_label = 'lr'
x = [item[x_label] for item in parsed_data]
y = [item[y_label] for item in parsed_data]

# Plotting
plt.plot(x, y, marker='o')
plt.title(x_label + ' vs ' + y_label)
plt.xlabel(x_label)
plt.ylabel(y_label)
plt.grid(True)
plt.show()