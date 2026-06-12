"""
make_post_image.py — render a branded "CPGuy" post image for a topic.

Image sourcing is confidence-ordered and never guesses:
  1. an explicit override image URL (you, replying with a link)
  2. the image shipped inside the RSS feed item (correct by construction)
  3. the og:image from the real article (Google News links are resolved first)
  4. text-only card if none of the above yield a verified image

First-time setup:
    pip install playwright requests
    playwright install chromium
"""

import os
import re
import html
import base64

import requests
from playwright.sync_api import sync_playwright

# ---- Brand (edit to taste) ----
WORDMARK_MAIN = "CPG"
WORDMARK_TAIL = "uy"
ACCENT = "#E2342D"
BG_TOP = "#C0392B"
BG_BOTTOM = "#1A0604"
WIDTH, HEIGHT = 1080, 1350

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) cpguy-bot/1.0"}


def _fetch_og_image(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=12, allow_redirects=True)
        if r.status_code != 200 or "text/html" not in r.headers.get("content-type", ""):
            return None
        for pat in (
            r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']',
            r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:image["\']',
            r'<meta[^>]+name=["\']twitter:image["\'][^>]+content=["\']([^"\']+)["\']',
        ):
            m = re.search(pat, r.text, re.I)
            if m:
                return m.group(1)
    except requests.RequestException:
        return None
    return None


def _image_data_uri(img_url):
    try:
        r = requests.get(img_url, headers=HEADERS, timeout=12)
        if r.status_code == 200 and r.content:
            ctype = r.headers.get("content-type", "image/jpeg").split(";")[0]
            return f"data:{ctype};base64," + base64.b64encode(r.content).decode()
    except requests.RequestException:
        return None
    return None


def _resolve_google_news(url):
    """Google News RSS links are encoded redirects. Open in a headless browser and
    follow to the real article URL. Non-Google-News URLs are returned unchanged."""
    if "news.google.com" not in url:
        return url
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(args=["--no-sandbox", "--disable-dev-shm-usage"])
            try:
                page = browser.new_context(user_agent=HEADERS["User-Agent"]).new_page()
                page.goto(url, wait_until="domcontentloaded", timeout=20000)
                try:
                    page.wait_for_url(lambda u: "news.google.com" not in u, timeout=15000)
                except Exception:
                    pass
                final = page.url
            finally:
                browser.close()
        return final if "news.google.com" not in final else None
    except Exception:
        return None


def _get_usable_image_uri(topic, override_url=None):
    if override_url:
        uri = _image_data_uri(override_url)
        if uri:
            return uri
    for img in (topic.get("images") or []):
        uri = _image_data_uri(img)
        if uri:
            return uri
    for link in (topic.get("links") or [])[:3]:
        real = _resolve_google_news(link)
        if not real:
            continue
        og = _fetch_og_image(real)
        if og:
            uri = _image_data_uri(og)
            if uri:
                return uri
    return None


def _headline_html(headline):
    words = headline.strip().upper().split()
    if not words:
        return ""
    first = f'<span class="accent">{html.escape(words[0])}</span>'
    rest = html.escape(" ".join(words[1:]))
    return first + ((" " + rest) if rest else "")


def _build_html(headline_html, image_uri):
    photo = (
        f"<div class=\"photo\" style=\"background-image:url('{image_uri}')\"></div>"
        '<div class="scrim"></div>'
        if image_uri else ""
    )
    return f"""<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Anton&family=Inter:wght@600;800&display=swap" rel="stylesheet">
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  html, body {{ width:{WIDTH}px; height:{HEIGHT}px; }}
  .card {{ position:relative; width:{WIDTH}px; height:{HEIGHT}px; overflow:hidden;
    background:linear-gradient(180deg, {BG_TOP} 0%, {BG_BOTTOM} 100%); }}
  .photo {{ position:absolute; inset:0; background-size:cover; background-position:center; }}
  .scrim {{ position:absolute; inset:0; background:linear-gradient(180deg,
      rgba(12,4,3,0.55) 0%, rgba(12,4,3,0.10) 18%, rgba(12,4,3,0) 42%,
      rgba(12,4,3,0.45) 66%, rgba(12,4,3,0.93) 100%); }}
  .content {{ position:absolute; inset:0; padding:72px; color:#fff;
    display:flex; flex-direction:column; justify-content:space-between;
    font-family:'Inter', sans-serif; }}
  .wordmark {{ font-size:46px; font-weight:800; letter-spacing:-1px;
    text-shadow:0 2px 12px rgba(0,0,0,0.45); }}
  .wordmark .tail {{ color:rgba(255,255,255,0.6); }}
  .divider {{ height:4px; width:140px; background:{ACCENT}; margin-bottom:26px; }}
  .headline {{ font-family:'Anton', sans-serif; font-size:80px; line-height:0.98;
    text-transform:uppercase; text-shadow:0 2px 16px rgba(0,0,0,0.55); }}
  .headline .accent {{ color:{ACCENT}; }}
</style></head>
<body><div class="card">
  {photo}
  <div class="content">
    <div class="wordmark">{WORDMARK_MAIN}<span class="tail">{WORDMARK_TAIL}</span></div>
    <div class="bottom">
      <div class="divider"></div>
      <div class="headline">{headline_html}</div>
    </div>
  </div>
</div></body></html>"""


def render_post_image(topic, out_path, override_url=None):
    """Render the post image for a topic. Returns (path, used_image: bool)."""
    image_uri = _get_usable_image_uri(topic, override_url)
    page_html = _build_html(_headline_html(topic.get("headline", "")), image_uri)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": WIDTH, "height": HEIGHT}, device_scale_factor=2)
        page.set_content(page_html, wait_until="networkidle", timeout=15000)
        page.wait_for_timeout(300)
        page.screenshot(path=out_path)
        browser.close()
    return out_path, bool(image_uri)
