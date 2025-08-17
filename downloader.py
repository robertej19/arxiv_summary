# pip install arxiv==2.1.0 tenacity tqdm
import arxiv, os, re, json, hashlib
from tenacity import retry, wait_fixed, stop_after_attempt, wait_exponential
from tqdm import tqdm
import time

OUT = "corpus_arxiv_rl"
os.makedirs(f"{OUT}/pdf", exist_ok=True)
os.makedirs(f"{OUT}/meta", exist_ok=True)

# Simplified query to avoid API issues
QUERY = "(" \
  "ti:\"reinforcement learning\" OR abs:\"reinforcement learning\" " \
  "OR ti:\"policy gradient\" OR abs:\"policy gradient\" " \
  "OR ti:PPO OR abs:PPO OR ti:TRPO OR abs:TRPO " \
  "OR ti:DDPG OR abs:DDPG OR ti:SAC OR abs:SAC " \
  "OR ti:TD3 OR abs:TD3 OR ti:DQN OR abs:DQN" \
  ") AND (cat:cs.LG OR cat:cs.AI OR cat:stat.ML) " \
  "AND submittedDate:[2018-01-01 TO 2022-12-31]"

# Fallback query if the main query fails
FALLBACK_QUERY = "ti:\"reinforcement learning\" AND cat:cs.LG AND submittedDate:[2018-01-01 TO 2022-12-31]"

@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3))
def safe_search(query, max_results=100):
    """Safely perform arXiv search with retry logic"""
    try:
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )
        return list(search.results())
    except Exception as e:
        print(f"Search failed with error: {e}")
        print(f"Retrying search for max_results={max_results}")
        time.sleep(2)
        raise

@retry(wait=wait_fixed(2), stop=stop_after_attempt(5))
def safe_download(result, path):
    result.download_pdf(filename=path)

def fetch_papers_in_batches(batch_size=100, max_total=2000):
    """Fetch papers in batches to avoid API issues"""
    all_results = []
    current_query = QUERY
    
    with tqdm(total=max_total, desc="Fetching papers") as pbar:
        while len(all_results) < max_total:
            try:
                # Calculate how many results we still need
                remaining = max_total - len(all_results)
                current_batch_size = min(batch_size, remaining)
                
                batch = safe_search(current_query, max_results=current_batch_size)
                if not batch:
                    print(f"No more results found")
                    break
                
                all_results.extend(batch)
                pbar.update(len(batch))
                
                # Small delay between batches to be respectful to the API
                time.sleep(1)
                
                # If we got fewer results than requested, we've reached the end
                if len(batch) < current_batch_size:
                    break
                    
            except Exception as e:
                print(f"Failed to fetch batch: {e}")
                if len(all_results) == 0:
                    # If we can't even get the first batch, try with a simpler query
                    print("Trying with simplified query...")
                    current_query = FALLBACK_QUERY
                    continue
                else:
                    # If we got some results but failed later, break
                    break
    
    return all_results

# Main execution
print("Starting arXiv paper collection...")
print(f"Query: {QUERY}")

try:
    results = fetch_papers_in_batches()
    print(f"Successfully fetched {len(results)} papers")
except Exception as e:
    print(f"Failed to fetch papers: {e}")
    exit(1)

rows = []
for r in tqdm(results, desc="Processing papers"):
    # lightweight keyword filter to keep things tight
    text = f"{r.title} || {r.summary}".lower()
    if "survey" in text or "systematic review" in text:
        continue
    if not any(k in text for k in [
        "reinforcement learning","policy gradient","actor-critic","q-learning",
        "ppo","trpo","ddpg","sac","td3","a3c","a2c","dqn","offline rl","model-based"
    ]):
        continue

    arxid = r.get_short_id()  # e.g., 1901.00001v1
    pdf_path = f"{OUT}/pdf/{arxid}.pdf"
    meta_path = f"{OUT}/meta/{arxid}.json"

    if not os.path.exists(pdf_path):
        try:
            safe_download(r, pdf_path)
        except Exception as e:
            print(f"Failed to download {arxid}: {e}")
            continue

    meta = {
        "arxiv_id": arxid,
        "title": r.title,
        "abstract": r.summary,
        "primary_category": r.primary_category,
        "categories": list(r.categories),
        "authors": [a.name for a in r.authors],
        "published": r.published.strftime("%Y-%m-%d"),
        "updated": r.updated.strftime("%Y-%m-%d"),
        "pdf_path": pdf_path,
        "doi": getattr(r, "doi", None),
        "comment": getattr(r, "comment", None),
        "journal_ref": getattr(r, "journal_ref", None),
        "links": [str(l.href) for l in r.links],
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    rows.append(meta)

print(f"Kept {len(rows)} candidates")
print(f"Total papers processed: {len(results)}")
