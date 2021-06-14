import numpy as np
import pandas as pd
import argparse
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder

parser = argparse.ArgumentParser()
parser.add_argument("-i", type=str, help="ACM file")
parser.add_argument("-o", type=str, help="cleaned file")
args = parser.parse_args()

content, n, m = [], 0, 0
with open(args.i, 'r') as f:
    for line in f.readlines():
        (ua, oa, p) = line.split(';')
        ua = ua.strip().split(',')
        oa = oa.strip().split(',')
        p = [p.strip()]
        line = ua + oa + p
        content.append(line)
        n, m = len(ua), len(oa)

df = pd.DataFrame(content)
df = df.apply(LabelEncoder().fit_transform)

cols = list(df.columns)
ua_list = df[cols[:n]].values.tolist()
oa_list = df[cols[n:n+m]].values.tolist()
p_list = df[cols[-1]].values.tolist()

rows = []
f = lambda l : [str(x) for x in l]
for (u, o, p) in zip(ua_list, oa_list, p_list):
    row = f"{','.join(f(u))};{','.join(f(o))};{str(p)}"
    rows.append(row)

with open(args.o, 'w') as f:
    f.write('\n'.join(rows))