import json
from pathlib import Path
import pandas as pd
import yaml

def get_config():
    return yaml.safe_load(Path("../config.yaml").read_text())


def get_data(path):
    return pd.read_feather("../" + path)

def main():
    config = get_config()
    data = get_data(config['paths']['calibration_raw'])
    with open("../" + config['paths']['calibration'], 'w', encoding="UTF-8") as f:
        for idx, row in data.iterrows():
            d = {
                'query_id':str(row['query_id']),
                'query_text':row['query_text'],
                'ground_truth':str(row['ground_truth']).split()
            }
            f.write(json.dumps(d, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    main()

