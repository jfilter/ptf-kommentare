import pandas as pd
import json
from tqdm import tqdm

lines = []
with open('data_run2.json') as f:
    for line in tqdm(f):
        d = json.loads(line)
        if 'url' in d:
            lines.append({'url': d['url'], {'num': len(d['comments']})
df = pd.DataFrame(lines)
df.to_csv('urls.csv', header=False, index=False)
