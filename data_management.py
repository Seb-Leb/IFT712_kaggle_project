def parse_csv(file_path):
    data = []
    with open('data.csv', 'r') as f:
        for n,l in enumerate(f):
            ls = l.strip().split(',')
            if n==0:
                keys = [x.replace('"','') for x in ls]
                features = keys[2:-1]
                continue
            line = dict(zip(keys, ls))
            for f in features:
                line[f] = float(line[f])
            data.append(line)
    return data