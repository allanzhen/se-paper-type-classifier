import requests
import pandas as pd
import time

QUERIES = [
    "technical debt",
    "architectural debt",
    "code debt",
    "design debt",
    "test debt",
    "documentation debt",
]

def fetch_papers(query, offset):
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "offset": offset,
        "limit": 100,
        "fields": "paperId,title,abstract,year,venue"
    }
    response = requests.get(url, params=params)
    
    if response.status_code == 429:
        print("  Rate limited — waiting 60 seconds...")
        time.sleep(60)
        response = requests.get(url, params=params)
    
    if response.status_code != 200:
        print(f"  Error: status code {response.status_code}")
        print(f"  Response: {response.text[:200]}")
        return []
    
    time.sleep(3)
    return response.json().get("data", [])

all_papers = []
seen_ids = set()

for query in QUERIES:
    print(f"Searching: {query}")
    for page in range(3):
        offset = page * 100
        papers = fetch_papers(query, offset)

        if not papers:
            break

        for paper in papers:
            pid = paper.get("paperId")
            if pid and pid not in seen_ids:
                seen_ids.add(pid)
                all_papers.append(paper)

        print(f"  Page {page + 1}: total so far = {len(all_papers)}")

        if len(papers) < 100:
            break

rows = []
for p in all_papers:
    rows.append({
        "paper_id": p.get("paperId", ""),
        "title":    p.get("title", ""),
        "abstract": p.get("abstract", ""),
        "year":     p.get("year", ""),
        "venue":    p.get("venue", ""),
    })

df = pd.DataFrame(rows)
df = df[df["abstract"].notna() & (df["abstract"].str.len() > 100)]
df = df.drop_duplicates(subset=["paper_id"])
df.to_csv("../../data/raw/technical_debt_corpus.csv", index=False)

print(f"Done! Saved {len(df)} papers")