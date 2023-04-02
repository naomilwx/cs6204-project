import re
import numpy as np

def get_printout_value(line, key):
    parts = line.split(' | ')
    for p in parts:
        k, v = parse_key_value(p)
        if k == key.strip():
            return v
    return None

def parse_key_value(part):
    matches = re.search("([A-Za-z0-9 ]+) (\d*\.?\d+|\d+)", part)
    k, v = matches.groups()
    return k.strip(), float(v)

def get_values(lines):
    values = {}
    for l in lines:
        parts = l.split(' | ')
        for p in parts:
            k, v = parse_key_value(p)
            if k == 'Episode':
                continue
            if k not in values:
                values[k] = []
            values[k].append(v)
    return values

def get_standard_devs(values):
    return {k: np.array(v).std() for k, v in values.items()}

def get_average(values):
    return {k: np.array(v).mean() for k, v in values.items()}
