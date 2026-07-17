import pandas as pd
import yaml
from pathlib import Path

def get_chunk_articles(path):
    df = pd.read_parquet(path)
    return dict(zip(df['chunk_id'].astype(str), df['article_id'].astype(str)))


def aggregate_scores(chunk_scores, chunk_articles, mode, top_k):
    per_article = {}
    for chink_id, score in chunk_scores:
        article_id = chunk_articles[chink_id]
        if article_id not in per_article:
            per_article[article_id] = []
        per_article[article_id].append(score)
    
    aggregated = []
    for article_id, score in per_article.items():
        if mode == 'max':
            aggregated.append((article_id, max(score)))
        if mode == 'mean_topk':
            best = sorted(score, reverse=True)[:top_k]
            aggregated.append((article_id, sum(best) / len(best)))
    aggregated.sort(key=lambda x: x[1], reverse=True)
    return aggregated





