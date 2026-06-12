"""
digest_bot.py — email your daily CPG digest, then generate posts from your reply.

Two commands:
    python digest_bot.py morning   # run pipeline, save topics, email the top-5 digest
    python digest_bot.py replies   # check inbox; for each reply, email back IG + TikTok posts

Local workflow:
    1. python digest_bot.py morning      -> you get the digest email
    2. reply to that email with a number (1-5)
    3. python digest_bot.py replies      -> you get an email with the posts

Setup — add these to your .env (alongside ANTHROPIC_API_KEY):
    EMAIL_ADDRESS=you@gmail.com
    EMAIL_APP_PASSWORD=abcd efgh ijkl mnop   # Gmail APP PASSWORD, not your login password
    EMAIL_TO=you@gmail.com                    # where the digest goes (can be the same)

Gmail notes:
    - Turn on 2-Step Verification, then create an App Password at
      https://myaccount.google.com/apppasswords and paste it above.
    - Make sure IMAP is enabled: Gmail Settings -> Forwarding and POP/IMAP -> Enable IMAP.
"""

import os
import sys
import re
import json
import ssl
import smtplib
import imaplib
import email
from datetime import datetime
from email.message import EmailMessage

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from make_post_image import render_post_image
import anthropic

# Reuse the pipeline you already built (importing also applies its SSL/.env setup)
from trending_news_agent import (
    FEEDS, collect_articles, embed, cluster, rank_clusters, summarize,
)

load_dotenv()

EMAIL_ADDRESS = os.environ.get("EMAIL_ADDRESS")
EMAIL_APP_PASSWORD = os.environ.get("EMAIL_APP_PASSWORD")
EMAIL_TO = os.environ.get("EMAIL_TO", EMAIL_ADDRESS)

SMTP_HOST, SMTP_PORT = "smtp.gmail.com", 465
IMAP_HOST = "imap.gmail.com"
STORE_DIR = "digests"
SUBJECT_TAG = "[CPG digest]"
POST_MODEL = "claude-sonnet-4-6"  # Sonnet writes better social copy; swap to haiku to save cost


# ---------------- Pipeline ----------------
def run_pipeline():
    articles = collect_articles(FEEDS)
    if not articles:
        return []
    model = SentenceTransformer("all-MiniLM-L6-v2")
    labels = cluster(embed(articles, model))
    top = rank_clusters(articles, labels)
    summaries = summarize(top)
    topics = []
    for i, (s, c) in enumerate(zip(summaries, top), 1):
        topics.append({
            "id": i,
            "headline": s.get("headline", ""),
            "summary": s.get("summary", ""),
            "why_trending": s.get("why_trending", ""),
            "sources": sorted({a["source"] for a in c["articles"]}),
            "links": [a["link"] for a in c["articles"][:5]],
            "titles": [a["title"] for a in c["articles"][:8]],
            "images": [a["image"] for a in c["articles"] if a.get("image")][:5],
        })
    return topics


# ---------------- Storage ----------------
def _today_key():
    return datetime.now().strftime("%Y-%m-%d")

def save_digest(topics):
    os.makedirs(STORE_DIR, exist_ok=True)
    path = os.path.join(STORE_DIR, f"{_today_key()}.json")
    with open(path, "w") as f:
        json.dump({"date": _today_key(), "topics": topics}, f, indent=2)
    return path

def load_latest_digest():
    if not os.path.isdir(STORE_DIR):
        return None
    files = sorted(f for f in os.listdir(STORE_DIR) if f.endswith(".json"))
    if not files:
        return None
    with open(os.path.join(STORE_DIR, files[-1])) as f:
        return json.load(f)


# ---------------- Email ----------------
def send_email(subject, body, attachments=None):
    import mimetypes
    msg = EmailMessage()
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = EMAIL_TO
    msg["Subject"] = subject
    msg.set_content(body)
    for path in (attachments or []):
        ctype, _ = mimetypes.guess_type(path)
        maintype, subtype = (ctype or "application/octet-stream").split("/", 1)
        with open(path, "rb") as f:
            msg.add_attachment(f.read(), maintype=maintype, subtype=subtype,
                               filename=os.path.basename(path))
    ctx = ssl.create_default_context()
    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=ctx) as smtp:
        smtp.login(EMAIL_ADDRESS, EMAIL_APP_PASSWORD)
        smtp.send_message(msg)

def format_digest(topics):
    lines = ["Today's top CPG topics. Reply with a number (1-5) to get an Instagram + TikTok post.", ""]
    for t in topics:
        lines += [
            f"{t['id']}. {t['headline']}",
            f"   {t['summary']}",
            f"   Why trending: {t['why_trending']}",
            f"   Sources: {', '.join(t['sources'])}",
            "",
        ]
    lines.append("Reply with just the number of the topic you want posts for.")
    return "\n".join(lines)

def _get_body(msg):
    if msg.is_multipart():
        for part in msg.walk():
            disp = str(part.get("Content-Disposition"))
            if part.get_content_type() == "text/plain" and "attachment" not in disp:
                payload = part.get_payload(decode=True)
                if payload:
                    return payload.decode(part.get_content_charset() or "utf-8", "ignore")
        return ""
    payload = msg.get_payload(decode=True)
    return payload.decode(msg.get_content_charset() or "utf-8", "ignore") if payload else ""

def _extract_choice(body):
    for raw in body.splitlines():
        line = raw.strip()
        if line.startswith(">") or (line.lower().startswith("on ") and "wrote:" in line.lower()):
            break
        if not line:
            continue
        cleaned = re.sub(r"https?://\S+", "", line)  # don't read digits out of a URL
        m = re.search(r"\b([1-5])\b", cleaned)
        if m:
            return int(m.group(1))
    return None

def _extract_image_url(body):
    for raw in body.splitlines():
        line = raw.strip()
        if line.startswith(">") or (line.lower().startswith("on ") and "wrote:" in line.lower()):
            break
        m = re.search(r"https?://\S+", line)
        if m:
            return m.group(0).rstrip(".,)")
    return None


# ---------------- Post generation ----------------
def generate_posts(topic):
    context = (
        f"Headline: {topic['headline']}\n"
        f"Summary: {topic['summary']}\n"
        f"Why it's trending: {topic['why_trending']}\n"
        "Source headlines:\n" + "\n".join(f"- {t}" for t in topic["titles"])
    )
    prompt = (
        "You are a social media writer for a CPG (consumer packaged goods) news brand, "
        "in the punchy, fast-paced style of accounts like CPG Wire. Using only the story "
        "below, write two things:\n\n"
        "1) INSTAGRAM POST: a scroll-stopping caption — strong hook on line one, 2-4 short "
        "lines of context, a closing line — plus 8-12 relevant hashtags.\n"
        "2) TIKTOK SCRIPT: a 25-40 second script with a hook line, 3-5 spoken beats, a "
        "suggested on-screen text caption for each beat, and a closing call to action.\n\n"
        "Stay factual to the story; don't invent figures. Output plain text with clear "
        "'=== INSTAGRAM ===' and '=== TIKTOK ===' section headers.\n\n"
        f"STORY:\n{context}"
    )
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=POST_MODEL,
        max_tokens=1200,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text


# ---------------- Commands ----------------
def cmd_morning():
    print("Running pipeline...")
    topics = run_pipeline()
    if not topics:
        print("No topics today — check your feeds.")
        return
    path = save_digest(topics)
    send_email(f"{SUBJECT_TAG} {_today_key()}", format_digest(topics))
    print(f"Sent digest with {len(topics)} topics. Saved to {path}")

def cmd_replies():
    digest = load_latest_digest()
    if not digest:
        print("No saved digest yet — run 'morning' first.")
        return
    topics = {t["id"]: t for t in digest["topics"]}

    imap = imaplib.IMAP4_SSL(IMAP_HOST)
    imap.login(EMAIL_ADDRESS, EMAIL_APP_PASSWORD)
    imap.select("INBOX")
    status, data = imap.search(None, f'(UNSEEN SUBJECT "{SUBJECT_TAG.strip("[]")}")')
    ids = data[0].split()
    if not ids:
        print("No new replies.")
        imap.logout()
        return

    for num in ids:
        status, msg_data = imap.fetch(num, "(RFC822)")
        msg = email.message_from_bytes(msg_data[0][1])
        if "re:" not in (msg["Subject"] or "").lower():
            continue
        body = _get_body(msg)
        choice = _extract_choice(body)
        override = _extract_image_url(body)        # optional forced image
        if choice and choice in topics:
            topic = topics[choice]
            posts = generate_posts(topic)
            img_path = os.path.join("posts", f"{digest['date']}-{choice}.png")
            try:
                _, used_img = render_post_image(topic, img_path, override_url=override)
                send_email(f"Posts — {topic['headline']}", posts, attachments=[img_path])
                src = "override" if override else ("article image" if used_img else "text-only")
                print(f"Sent posts + image for #{choice} ({src})")
            except Exception as e:
                send_email(f"Posts — {topic['headline']}", posts)
                print(f"Sent posts for #{choice}; image render failed: {e}")
        else:
            print(f"Couldn't read a 1-5 choice from a reply (got {choice}).")
    imap.logout()


def _check_env():
    missing = [k for k in ("ANTHROPIC_API_KEY", "EMAIL_ADDRESS", "EMAIL_APP_PASSWORD")
               if not os.environ.get(k)]
    if missing:
        print("Missing env vars in .env:", ", ".join(missing))
        sys.exit(1)


if __name__ == "__main__":
    _check_env()
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "morning":
        cmd_morning()
    elif cmd == "replies":
        cmd_replies()
    else:
        print("Usage: python digest_bot.py [morning|replies]")
