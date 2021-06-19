import argparse
from config import cat_dict

parser = argparse.ArgumentParser()
parser.add_argument("-i", type=str, help="ACM file")
parser.add_argument("-o", type=str, help="cleaned file")
args = parser.parse_args()

content = []
with open(args.i, 'r') as f:
    for line in f.readlines():
        (ua, oa, p) = line.split(';')
        ua = [str(cat_dict[x]) for x in ua.strip().split(',')]
        oa = [str(cat_dict[x]) for x in oa.strip().split(',')]
        line = f"{','.join(ua)};{','.join(oa)};{p.strip()}"
        content.append(line)

with open(f"./ML/{args.o}", 'w') as f:
    f.write('\n'.join(content))