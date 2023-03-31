import re
def get_printout_value(line, key):
    parts = line.split(' | ')
    for p in parts:
        matches = re.search("([A-Za-z0-9 ]+) (\d*\.?\d+|\d+)", p)
        k, v = matches.groups()
        if k.strip() == key.strip():
            return float(v)
    return None
