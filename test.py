with open('data_utils/stable_objects_list.txt', 'r') as file:
    lines = file.readlines()

result = []
for line in lines:
    if len(line) > 59:  # Ensure line is long enough
        result.append(line[42:-18])

print(result)