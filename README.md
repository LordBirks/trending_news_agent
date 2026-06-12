# Trending Consumer News Agent

A pipeline that collects consumer news, clusters related stories, ranks them by
how "trending" they are, and summarizes the top topics with Claude.

## Pipeline
collect (RSS) -> dedup -> embed -> cluster -> rank -> summarize (Claude)

- **collect** — pull recent articles from RSS feeds (see `FEEDS` in the script)
- **embed** — sentence-transformers embeddings of title + summary
- **cluster** — agglomerative clustering groups articles about the same story
- **rank** — score = source diversity + volume + recency; keep top N
- **summarize** — Claude writes headline / summary / why-trending per cluster

## Setup
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # then paste your ANTHROPIC_API_KEY
python trending_news_agent.py
```

## Tuning knobs (top of trending_news_agent.py)
- `FEEDS` — your sources
- `HOURS_LOOKBACK` — recency window
- `DISTANCE_THRESHOLD` — lower = stricter story grouping
- `TOP_N` — how many trending topics to surface
- ranking weights in `rank_clusters()` — what "trending" means to you

## Roadmap / open forks
- [ ] quick digest vs. production service (add SQLite store + scheduling)
- [ ] trending definition: volume vs. velocity vs. social engagement
- [ ] output target: console -> email / Slack / stored feed
- [ ] agentic version: give Claude tools (search/fetch) to choose sources
- [ ] add a news API (NewsAPI/GNews) and Reddit/HN signal alongside RSS
# trending_news_agent
