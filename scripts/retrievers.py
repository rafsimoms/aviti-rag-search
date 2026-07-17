import pickle
import re
import pandas as pd
import pymorphy3
from rank_bm25 import BM25Okapi

from aggregate import get_chunk_articles, aggregate_scores

 
morph = pymorphy3.MorphAnalyzer()
lemma_cache = {}
token_re = re.compile(r"[а-яёa-z0-9]+")

def tokenize(text):
    tokens = token_re.findall(text.lower())
    lemmas = []
    for token in tokens:
        if token not in lemma_cache:
            lemma_cache[token] = morph.parse(token)[0].normal_form
        lemmas.append(lemma_cache[token])
    return lemmas

class BaseRetriever:
    def __init__(self, config):
        self.config = config
        self.chunk_articles = get_chunk_articles(config['paths']['processed_articles'])
    
    def fit(self, df):
        raise NotImplementedError
 
    def _search_chunks(self, query):
        raise NotImplementedError
    
    def search(self, query):
        chunk_scores = self._search_chunks(query)
        aggregated = aggregate_scores(chunk_scores=chunk_scores, chunk_articles=self.chunk_articles, 
                                      mode=self.config['search']['aggregation'],
                                      top_k=self.config['search']['agg_top_k'])
        return aggregated[:self.config['search']['top_k']]


class BM25Retriever(BaseRetriever):
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({'bm25': self.bm25, 'chunk_ids': self.chunk_ids}, f)
 
    def load(self, path):
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.bm25 = state['bm25']
        self.chunk_ids = state['chunk_ids']
    
    def fit(self, df):
        self.chunk_ids = df['chunk_id'].astype(str).tolist()
        corpus = [tokenize(text) for text in df['chunk']]
        self.bm25 = BM25Okapi(corpus)
    
    def _search_chunks(self, query):
        scores = self.bm25.get_scores(tokenize(query))
        pairs = list(zip(self.chunk_ids, scores))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs[:self.config['search']['candidate_chunks']]
    



    

