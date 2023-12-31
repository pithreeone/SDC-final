import json

# File path of the JSON file
file_path = "/home/pithreeone/SDC-Repo/2023_final/data/mini_test/city_7_0/gt_city_7_0.json"

# Read data from the JSON file
with open(file_path, 'r') as json_file:
    data = json.load(json_file)

new_data = []
for temp in data:
    temp["score"] = 1.0
    new_data.append(temp)
    # print(temp)

json_file = "new_gt.json"
with open(json_file, 'w') as json_file:
    json.dump(data, json_file, indent=2)  # indent for pretty formatting (optional)  

# Now, 'data' contains the content of the JSON file
# print(data)