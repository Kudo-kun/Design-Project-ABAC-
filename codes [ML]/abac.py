from csv import reader
import numpy as np
from time import time

with open("sub_data.csv", 'r') as f:
    u_reader = reader(f)
    sub_data = list(u_reader)
    f.close()

with open("obj_data.csv", 'r') as f:
    o_reader = reader(f)
    obj_data = list(o_reader)
    f.close()

with open("pol_data.csv", 'r') as f:
    r_reader = reader(f)
    pol_data = list(r_reader)
    f.close()

def eval_rule(rule, so_pair):
    for i in range(len(rule)):
        if (rule[i] != 'X') and (rule[i] != so_pair[i]):
            return False
    return True

start = time()
training_data = []
acm = np.zeros((len(sub_data)-1, len(obj_data)-1))

for i in range(1, len(sub_data)):
    for j in range(1, len(obj_data)):
        so_pair = (sub_data[i] + obj_data[j])
        for k in range (1, len(pol_data)):
            if eval_rule(pol_data[k], so_pair):
                acm[i-1][j-1] = 1
                break

print(np.sum(acm))
for i in range(1, len(sub_data)):
    for j in range(1, len(obj_data)):
        temp_sub = ','.join(sub_data[i])
        temp_obj = ','.join(obj_data[j])
        curr_row = "{};{};{}".format(temp_sub, temp_obj, int(acm[i-1][j-1]))
        training_data.append(curr_row)

training_data = list(set(training_data))
with open("ACM.txt", 'w') as f:
    f.write('\n'.join(training_data))
    f.close()

print(f"Runtime of the program is {time() - start}")