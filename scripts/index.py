import pandas as pd
import argparse
from retrievers import BM25Retriever, DenseRetriever
import yaml
from pathlib import Path

def load_config(path):
    return yaml.safe_load(Path(path).read_text())

def build_bm25(config, df):
    r = BM25Retriever(config)
    r.fit(df)
    r.save('../' +config['paths']['bm25_index'])

def build_dense(config, df):
    r = DenseRetriever(config)
    r.fit(df)
    r.save('../' +config['paths']['dense_index'])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--retriever', required=True, choices=['bm25', 'dence', 'all'])
    args = parser.parse_args()
    config = load_config('../config.yaml')
    df = pd.read_parquet('../' + config['paths']['processed_articles'])
    if args.retriever == 'bm25':
        build_bm25(config, df)
    elif args.retriever == 'dense':
        build_dense(config, df)
    elif args.retriever == 'all':
        build_bm25(config, df)
        build_dense(config, df)
    
if __name__ == '__main__':
    main()
    
