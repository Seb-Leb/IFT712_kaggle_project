def parse_csv(file_path):
    with open(file_path, 'r') as f:
        for n,l in enumerate(f):
            ls = l.strip().split(',')
            if n==0:
                keys = ls
                continue
            line = dict(zip(keys, ls))
            data.append(line)
    return data
