# pip install arxiv==2.1.0 tenacity tqdm
import arxiv, os, json, time, re
from tenacity import retry, wait_fixed, stop_after_attempt
from tqdm import tqdm

OUT = "corpus_arxiv_rl"
os.makedirs(f"{OUT}/pdf", exist_ok=True)
os.makedirs(f"{OUT}/meta", exist_ok=True)

YEARS = list(range(2018, 2023))  # 2018–2022 inclusive
CATS = ["cs.LG", "cs.AI", "stat.ML"]  # add "econ.EM" if you want

# Core RL query; we’ll AND it with year & category shards
BASE_Q = "(" \
  "ti:\"reinforcement learning\" OR abs:\"reinforcement learning\" " \
  "OR ti:\"policy gradient\" OR abs:\"policy gradient\" " \
  "OR ti:PPO OR abs:PPO OR ti:TRPO OR abs:TRPO OR ti:DDPG OR abs:DDPG " \
  "OR ti:SAC OR abs:SAC OR ti:TD3 OR abs:TD3 OR ti:DQN OR abs:DQN" \
  ")"

def build_query(cat, year):
    return f"({BASE_Q}) AND cat:{cat} AND submittedDate:[{year}-01-01 TO {year}-12-31]"

def make_client(delay=3, retries=5, page_size=50):
    # Smaller page_size + delay helps avoid empty pages / throttling
    return arxiv.Client(page_size=page_size, delay_seconds=delay, num_retries=retries)

client = make_client()

@retry(wait=wait_fixed(2), stop=stop_after_attempt(5))
def safe_download(result, path):
    result.download_pdf(filename=path)

def is_keep(title, abstract):
    t = f"{title} || {abstract}".lower()
    if "survey" in t or "systematic review" in t:
        return False
    # extra filter to stay RL-focused
    return any(k in t for k in [
        "reinforcement learning","policy gradient","actor-critic","q-learning",
        "ppo","trpo","ddpg","sac","td3","a3c","a2c","dqn","offline rl","model-based"
    ])

kept = 0
for year in YEARS:
    for cat in CATS:
        query = build_query(cat, year)
        search = arxiv.Search(
            query=query,
            max_results=1000,  # upper bound per shard; client paginates under the hood
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )

        while True:
            try:
                for r in tqdm(client.results(search), desc=f"{year}-{cat}"):
                    if not is_keep(r.title, r.summary):
                        continue

                    arxid = r.get_short_id()        # e.g. 1901.00001v1
                    base_id = arxid.split("v")[0]   # de-dupe later by keeping highest v
                    pdf_path = f"{OUT}/pdf/{arxid}.pdf"
                    meta_path = f"{OUT}/meta/{arxid}.json"

                    if not os.path.exists(pdf_path):
                        try:
                            safe_download(r, pdf_path)
                        except Exception as e:
                            print(f"Download failed {arxid}: {e}")
                            continue

                    meta = {
                        "arxiv_id": arxid,
                        "base_id": base_id,
                        "title": r.title,
                        "abstract": r.summary,
                        "categories": list(r.categories),
                        "primary_category": r.primary_category,
                        "authors": [a.name for a in r.authors],
                        "published": r.published.strftime("%Y-%m-%d"),
                        "updated": r.updated.strftime("%Y-%m-%d"),
                        "pdf_path": pdf_path,
                    }
                    with open(meta_path, "w") as f:
                        json.dump(meta, f, ensure_ascii=False, indent=2)
                    kept += 1

                break  # finished this shard without error

            except arxiv.UnexpectedEmptyPageError as e:
                # Back off and retry this shard with a slower client
                print(f"\n[warn] Empty page on {year}-{cat}. Backing off… {e}")
                time.sleep(5)
                client = make_client(delay=6, retries=8, page_size=25)
                # loop continues; we re-run the shard with the new client
            except Exception as e:
                print(f"\n[warn] Other error on {year}-{cat}: {e}")
                # brief pause then continue; you could also break to skip shard
                time.sleep(3)
                client = make_client(delay=6, retries=8, page_size=25)

print(f"Kept ~{kept} candidates (pre-dedupe)")
