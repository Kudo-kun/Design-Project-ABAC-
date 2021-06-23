from csv import writer

pos, neg = 0, 0
content = [["Designation", "Department", "Degree", "Year", "Type", "Department", "Degree", "Year"]]

with open("model_ABAC_test_data_set.csv", 'r') as f:
    for line in f.readlines()[1:]:
        line = line.strip()
        arr = line.split(',')
        uy = -1 if arr[3] == "NA" else int(arr[3])
        oy = -1 if arr[7] == "NA" else int(arr[7])
        if arr[-1] == '1' and (uy > oy):
            arr[-1] = '0'
        content.append(arr)
        if arr[-1] == '1':
            pos += 1
        else:
            neg += 1

print(f"pos = {pos}, neg = {neg}")

with open('model_ABAC_new_test_data_set.csv', 'w', newline='\n') as f:
    writer = writer(f)
    writer.writerows(content)
