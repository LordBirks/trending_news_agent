"""
Trending Consumer News Agent - v1 starter
Pipeline: collect (RSS) -> dedup -> embed -> cluster -> rank -> summarize (Claude)

Setup:
    pip install feedparser sentence-transformers scikit-learn numpy anthropic
    export ANTHROPIC_API_KEY=sk-ant-...

Run:
    python trending_news_agent.py

Notes:
    - Swap/extend FEEDS with your own consumer-news sources.
    - sentence-transformers downloads a small (~80MB) model on first run.
      To avoid the local model, replace embed() with an embedding API
      (e.g. Voyage AI) and keep the rest unchanged.
"""

import os
import re
import json
from datetime import datetime, timezone, timedelta
from email.utils import parsedate_to_datetime

import numpy as np
import feedparser
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
import anthropic

import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()

# ---------------- Config ----------------
FEEDS = [
    # Direct trade-pub feeds: real article URLs + feed images (reliable auto-sourcing)
    "https://www.bevnet.com/feed/",            # beverage brands
    "https://www.nosh.com/feed/",              # packaged food brands
    "https://www.fooddive.com/feeds/news/",    # food industry
    "https://www.retaildive.com/feeds/news/",  # retail / distribution
    "https://www.grocerydive.com/feeds/news/", # grocery
    "https://insider.fitt.co/feed/",           # Fitt Insider (wellness)
    # If any returns 0 in the per-feed diagnostic, fall back to Google News site search, e.g.:
    #   "https://news.google.com/rss/search?q=site:bevnet.com&hl=en-US&gl=US&ceid=US:en"
]

HOURS_LOOKBACK = 120      # only consider articles this recent
MIN_CLUSTER_SIZE = 1     
TOP_N = 5                # how many trending topics to surface
DISTANCE_THRESHOLD = 0.45  # lower = stricter grouping (cosine distance)
SUMMARY_MODEL = "claude-haiku-4-5-20251001"  # check docs.claude.com for current model names

# ---------------- Helpers ----------------
def _parse_date(entry):
    if entry.get("published_parsed"):
        return datetime(*entry["published_parsed"][:6], tzinfo=timezone.utc)
    for key in ("published", "updated"):
        if entry.get(key):
            try:
                dt = parsedate_to_datetime(entry[key])
                return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
            except (TypeError, ValueError):
                pass
    return None


def _clean(html):
    text = re.sub(r"<[^>]+>", "", html or "")
    return re.sub(r"\s+", " ", text).strip()[:500]

def _entry_image(entry):
    # Pull a lead image from common RSS media fields, if the feed provides one
    for mc in entry.get("media_content", []) or []:
        if mc.get("url"):
            return mc["url"]
    for mt in entry.get("media_thumbnail", []) or []:
        if mt.get("url"):
            return mt["url"]
    for enc in entry.get("enclosures", []) or []:
        if str(enc.get("type", "")).startswith("image") and enc.get("href"):
            return enc["href"]
    for link in entry.get("links", []) or []:
        if link.get("rel") == "enclosure" and str(link.get("type", "")).startswith("image"):
            return link.get("href")
    return None

# ---------------- 1. Collect ----------------
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) trending-news-agent/1.0"

def _entry_source(entry, feed_title):
    # Google News puts the actual publisher in entry.source.title
    src = entry.get("source")
    if isinstance(src, dict) and src.get("title"):
        return src["title"]
    return feed_title

def collect_articles(feeds):
    cutoff = datetime.now(timezone.utc) - timedelta(hours=HOURS_LOOKBACK)
    articles, seen_links = [], set()
    for url in feeds:
        parsed = feedparser.parse(url, agent=USER_AGENT)
        status = getattr(parsed, "status", "n/a")
        feed_title = parsed.feed.get("title", url)
        kept = 0
        for e in parsed.entries:
            link = e.get("link", "")
            if not link or link in seen_links:
                continue
            published = _parse_date(e)
            if published and published < cutoff:
                continue
            seen_links.add(link)
            articles.append({
                "title": e.get("title", "").strip(),
                "summary": _clean(e.get("summary", "")),
                "link": link,
                "source": _entry_source(e, feed_title),
                "published": published,
                "image": _entry_image(e),
            })
            kept += 1
        flag = "" if parsed.entries else "  <-- returned nothing"
        print(f"  [{status}] {len(parsed.entries):>3} entries, {kept:>3} kept  {feed_title[:40]}{flag}")
    return articles


# ---------------- 2. Embed ----------------
def embed(articles, model):
    texts = [f"{a['title']}. {a['summary']}" for a in articles]
    return model.encode(texts, normalize_embeddings=True)


# ---------------- 3. Cluster ----------------
def cluster(embeddings):
    if len(embeddings) < 2:
        return np.zeros(len(embeddings), dtype=int)
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=DISTANCE_THRESHOLD,
    )
    return clustering.fit_predict(embeddings)


# ---------------- 4. Rank ----------------
EVENT_KEYWORDS = ("launch", "debut", "funding", "raise", "raises", "raised", "fundraise","seed", "series a",
                  "series b", "acquir", "merger", "rollout", "retail", "expands",
                  "partnership", "investment")

def _event_boost(group):
    text = " ".join((a["title"] + " " + a["summary"]).lower() for a in group)
    return sum(1 for k in EVENT_KEYWORDS if k in text)

def rank_clusters(articles, labels):
    groups = {}
    for art, label in zip(articles, labels):
        groups.setdefault(label, []).append(art)
    ranked = []
    now = datetime.now(timezone.utc)
    for group in groups.values():
        if len(group) < MIN_CLUSTER_SIZE:
            continue
        n_sources = len({a["source"] for a in group})
        ages = [(now - a["published"]).total_seconds() / 3600
                for a in group if a["published"]]
        recency = 1.0 / (1.0 + (min(ages) if ages else HOURS_LOOKBACK))
        # recency-led, since strong CPG stories often run in a single outlet
        score = recency * 6 + n_sources + len(group) + _event_boost(group) * 2
        ranked.append({"articles": group, "n_sources": n_sources, "score": score})
    ranked.sort(key=lambda c: c["score"], reverse=True)
    return ranked[:TOP_N]


# ---------------- 5. Summarize ----------------
def summarize(clusters):
    if not clusters:
        return []
    blocks = []
    for i, c in enumerate(clusters, 1):
        lines = [f"- {a['title']} ({a['source']})" for a in c["articles"][:8]]
        blocks.append(
            f"Cluster {i} ({len(c['articles'])} articles, "
            f"{c['n_sources']} sources):\n" + "\n".join(lines)
        )

    prompt = (
        f"You are a consumer-news analyst. Below are {len(clusters)} clusters of "
        "related news articles. For each cluster, write a tight summary of the story.\n\n"
        "Return ONLY a JSON array (no prose, no code fences), one object per cluster "
        "in the same order, each with keys: "
        '"headline" (punchy, <12 words), "summary" (2-3 sentences), '
        '"why_trending" (one sentence).\n\n'
        + "\n\n".join(blocks)
    )

    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=SUMMARY_MODEL,
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = resp.content[0].text.strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return [{"headline": "Parse error", "summary": raw, "why_trending": ""}]


# ---------------- Orchestrate ----------------
def main():
    print("Collecting articles...")
    articles = collect_articles(FEEDS)
    print(f"  {len(articles)} recent articles")
    if not articles:
        print("No articles found - check your feeds / lookback window.")
        return

    print("Embedding & clustering...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    labels = cluster(embed(articles, model))
    top = rank_clusters(articles, labels)
    print(f"  {len(top)} trending clusters")

    print("Summarizing with Claude...\n")
    summaries = summarize(top)

    print("=" * 60)
    print(f"TOP {len(summaries)} TRENDING CONSUMER TOPICS")
    print(f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}")
    print("=" * 60)
    for i, (s, c) in enumerate(zip(summaries, top), 1):
        print(f"\n{i}. {s.get('headline', '')}")
        print(f"   {s.get('summary', '')}")
        print(f"   Why trending: {s.get('why_trending', '')}")
        print(f"   Coverage: {len(c['articles'])} articles / {c['n_sources']} sources")
        print(f"   e.g. {c['articles'][0]['link']}")


if __name__ == "__main__":
    main()
