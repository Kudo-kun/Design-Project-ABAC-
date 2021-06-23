import argparse
from config import cat_dict

parser = argparse.ArgumentParser()
parser.add_argument("-i", type=str, help="ACM file")
parser.add_argument("-o", type=str, help="cleaned file")
args = parser.parse_args()

content = []
fun = lambda lst : [str(cat_dict[a]) for a in lst]

with open(args.i, 'r') as f:
    lines = f.read().split('\n')[1:]

for line in lines:
    arr = line.strip().split(',')
    if len(line) > 0:
        ua, oa, p = arr[:4], arr[4:-1], arr[-1]
        line = f"{','.join(fun(ua))};{','.join(fun(oa))};{p}"
        content.append(line)

with open(f"./ML/{args.o}", 'w') as f:
    f.write('\n'.join(content))
    