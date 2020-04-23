import pandas as pd
import json
from tqdm import tqdm

lines = []
with open('data_run2.json') as f:
    for line in tqdm(f):
        d = json.loads(line)
        if 'more_url' in d:
            lines.append(d)
with open('missing_more.json') as f:
    for line in tqdm(f):
        d = json.loads(line)
        if 'more_url' in d:
            lines.append(d)
df = pd.DataFrame(lines)
df.to_csv('urls.csv', header=False, index=False)
