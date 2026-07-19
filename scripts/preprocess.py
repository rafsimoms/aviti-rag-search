from pathlib import Path
import re
import pandas as pd
import yaml
from bs4 import BeautifulSoup

def load_config(path):
    return yaml.safe_load(Path(path).read_text())

def html_to_text(html):
    soup = BeautifulSoup(html, "lxml")
    for tag in soup.find_all(['input', 'img', 'style', 'script', 'noscript']):
        tag.decompose()
    text = soup.get_text()
    return re.sub(r"\s+", " ", text).strip()

def chunk_text(text, chunk_size, overlap):
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    for st in range(0, len(words), step):
        chunk = words[st:st + chunk_size]
        if chunk:
            chunks.append(chunk)
        if st + chunk_size >= len(words):
            break
    return chunks

def run():
    config = load_config('../config.yaml')
    chunk_size = config['chunking']['chunk_size']
    overlap = config['chunking']['chunk_overlap']
    data = pd.read_feather("../" + config['paths']['raw'])
    rows = []
    for idx, row in data.iterrows():
        text = html_to_text(row['body'])
        title = str(row['title']).strip()
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        article_id = row['article_id']
        for i in range(len(chunks)):
            rows.append({
                "article_id": article_id,
                "chunk_id": f"{article_id}_{i + 1}",
                "chunk": f"{title}. {" ".join(chunks[i])}",
                "position": i + 1
            })
    df = pd.DataFrame(rows)
    out = Path('../' + config['paths']['processed_articles'])
    df.to_parquet(out)
    print(f"создано {len(df)} чанков")


if __name__ == '__main__':
    run()



