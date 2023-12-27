import json
import matplotlib.pyplot as plt

# Specify the path to your JSON file
json_file_path = '/home/pithreeone/SDC-Repo/2023_final/vehicle_detection/train_results/faster_rcnn_R_50_FPN_3x_good_and_bad_weather/test.json'

# Read the JSON file
with open(json_file_path, 'r') as file:
    # Load the JSON data
    data = file.read()

# Split the data into individual JSON objects
json_objects = data.strip().split('\n')

# Parse each JSON object
parsed_data = [json.loads(obj) for obj in json_objects]

# Now you can work with the parsed data
# for item in parsed_data:
#     print(item)


# Assuming parsed_data is your list of dictionaries from the JSON
iterations = [item['iteration'] for item in parsed_data]
loss_cls_values = [item['loss_rpn_cls'] for item in parsed_data]

# Plotting
plt.plot(iterations, loss_cls_values, marker='o')
plt.title('Iteration vs data_time')
plt.xlabel('Iteration')
plt.ylabel('data_time')
plt.grid(True)
plt.show()