import json

# File path of the JSON file
file_path = "/home/pithreeone/SDC-Repo/2023_final/vehicle_detection/output.json"

# Read data from the JSON file
with open(file_path, 'r') as json_file:
    data = json.load(json_file)

new_data = []
for temp in data:
    score = temp["scores"]
    del temp["scores"]
    temp['score'] = score
    token = temp['sample_token']
    token = token[:-4]
    temp['sample_token'] = token
    temp['name'] = 'car'
    new_data.append(temp)
    # print(temp)

json_file = "output_new.json"
with open(json_file, 'w') as json_file:
    json.dump(data, json_file, indent=2)  # indent for pretty formatting (optional)  

# Now, 'data' contains the content of the JSON file
# print(data)