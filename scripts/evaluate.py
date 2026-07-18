import pandas as pd
import yaml
from pathlib import Path
import ranx
import time
from ranx import Qrels, Run, evaluate as ranx_evaluate
import random
import json
from retrievers import BM25Retriever, DenseRetriever, HybridRetriever

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
    qrels = {}
    run = {}
    latencies = []
    for line in data:
        qrels[line['query_id']] = {str(art_id): 1 for art_id in line['ground_truth']}
        t0 = time.perf_counter()
        results = retriever.search(line["query_text"])
        latencies.append(time.perf_counter() - t0)
        run[line['query_id']] = {str(art_id): float(score) for art_id, score in results}
    qrels_ranx = Qrels(qrels)
    run_ranx = Run(run)
    map_at_10 = ranx_evaluate(qrels_ranx, run_ranx, "map@10")
    avg_latency_ms = 1000 * sum(latencies) / len(latencies)
    print(f"map10: {map_at_10}, latency: {avg_latency_ms}")
    return map_at_10
    
if __name__ == "__main__":
    config = load_config("../config.yaml")

    bm25 = BM25Retriever(config)
    bm25.load("../" + config["paths"]["bm25_index"])

    dense = DenseRetriever(config)
    dense.load(Path("../" + config["paths"]["dense_index"]))

    hybrid = HybridRetriever(config, [bm25, dense])
    evaluate_retriever(hybrid, "../" + config["paths"]["calibration"])
