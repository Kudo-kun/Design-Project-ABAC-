UA, OA = [], []
total_rules = 0
with open("ACM.txt", 'r') as f:
    for line in f.readlines():
        total_rules += 1
        (ua, oa, _) = line.split(';')
        for x in ua.split(','):
            if x.isnumeric():
                x = "year-{}".format(x)
            UA.append(x)
        for x in oa.split(','):
            if x.isnumeric():
                x = "year-{}".format(x)
            OA.append(x)
    f.close()
    
UAD = {k: v for v, k in enumerate(list(set(UA)))}
OAD = {k: v for v, k in enumerate(list(set(OA)))}

rows = []
pos = 0
fun = lambda x : ("year-{}".format(x) if x.isnumeric() else x)

with open("ACM.txt", 'r') as f:
    for line in f.readlines():
        (ua, oa, p) = line.split(';')
        pos += (1 if p == "1\n" else 0)
        ua = [str(UAD[fun(x)] + 1) for x in ua.split(',')]
        oa = [str(OAD[fun(x)] + 1) for x in oa.split(',')]
        row = "{};{};{}".format(','.join(ua), ','.join(oa), p)
        rows.append(row)
    f.close()

with open("final_data.txt", 'w') as f:
    f.write(''.join(rows))
    f.close()

neg = total_rules-pos
print(pos, neg)