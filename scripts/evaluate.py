import pandas as pd
import yaml
from pathlib import Path
import ranx
import time
from ranx import Qrels, Run, evaluate as ranx_evaluate
import random
import json
from datetime import datetime
from retrievers import BM25Retriever, DenseRetriever, HybridRetriever, RerankRetriever, RandomRetriever
from tqdm.auto import tqdm
import argparse

def load_config(path):
    return yaml.safe_load(Path(path).read_text())

def get_data(path):
    data = []
    with open(path, 'r', encoding="UTF-8") as f:
        for line in f:
            line = line.rstrip()
            data.append(json.loads(line))
    return data

def evaluate_retriever(retriever, path):
    data = get_data(path)
    retriever.search(data[0]["query_text"])
    qrels = {}
    run = {}
    latencies = []
    for line in tqdm(data, desc='eval'):
        qrels[line['query_id']] = {str(art_id): 1 for art_id in line['ground_truth']}
        t0 = time.perf_counter()
        results = retriever.search(line["query_text"])
        latencies.append(time.perf_counter() - t0)
        run[line['query_id']] = {str(art_id): float(score) for art_id, score in results}
    qrels_ranx = Qrels(qrels)
    run_ranx = Run(run)
    map_at_10 = ranx_evaluate(qrels_ranx, run_ranx, "map@10")
    avg_latency_ms = 1000 * sum(latencies) / len(latencies)
    return (map_at_10, avg_latency_ms)

def build_retriver(name, config):
    if name == 'random':
        df = pd.read_parquet('../' + config['paths']['processed_acricles'])['erticle_id'].astype(str).tolist()
        r = RandomRetriever(article_ids=df, top_k=config['search']['top_k'])
        return r
    if name == 'bm25':
        r = BM25Retriever(config=config)
        r.load('../' + config['paths']['bm25_index'])
        return r
    if name == 'dense':
        r = DenseRetriever(config=config)
        r.load('../' + config['paths']['dense_index'])
        return r
    if name == 'hybrid':
        r = HybridRetriever(config, [build_retriver('bm25', config), build_retriver('dense', config)])
        return r
    if name == 'reranker':
        r = RerankRetriever(config, build_retriver('hybrid', config))
        return r
    raise ValueError(f"Такого ретривера не существует: {name}")

def log_experiment(config, retriever_name, map_at_10, avg_latency_ms):
    block = '\n'.join(['=' * 60,
                       f"time: {datetime.now().isoformat(timespec='seconds')}", 
                       f"retriever_name: {retriever_name}",
                       f'map@10: {map_at_10}',
                       f"latency: {avg_latency_ms}",
                       f"chunk_size: {config['chunking']['chunk_size']}",
                       f"overlap: {config['chunking']['chunk_overlap']}",
                       f"search cfg: ",
                       yaml.dump(config['search'], allow_unicode=True, sort_keys=False).rstrip(),
                       '=' * 60
                       ])
    with open('../' + config['paths']['logs'], 'a', encoding='UTF-8') as f:
        f.write(block + '\n')


def main():
    config = load_config("../config.yaml")
    parser = argparse.ArgumentParser()
    parser.add_argument('--retriever', required=True, choices=['random', 'bm25', 'dense', 'hybrid', 'reranker'])
    args = parser.parse_args()
    retriever = build_retriver(args.retriever, config=config)
    map_at_10, avg_latency_ms = evaluate_retriever(retriever=retriever, path='../' + config['paths']['calibration'])
    print(f"map10: {map_at_10}, latency: {avg_latency_ms}")
    log_experiment(config=config, retriever_name=args.retriever, map_at_10=map_at_10, avg_latency_ms=avg_latency_ms)



if __name__ == "__main__":
    main()


