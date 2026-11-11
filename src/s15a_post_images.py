"""Stage 15a (Jev): one still image per Study 2 target post, so Jev sees what the post shows
(91% of the 2025 top posts are images/videos with a short title).

image post -> preview source; video -> preview frame; gallery -> first media_metadata item.
Resized to <= 768 px on the long side (Jev accepts 280-1568 px per side; ~1.1k tokens).

Run with STUDY=global."""
import html as htmllib
import io
import os
import re

import requests
from PIL import Image

from api import DATA, ROOT, get

OUT = os.path.join(DATA, "post_images")
os.makedirs(OUT, exist_ok=True)

page = open(os.path.join(ROOT, "data", "Top Posts of 2025.html"), encoding="utf-8").read()
ids = list(dict.fromkeys(re.findall(r"/r/[A-Za-z0-9_]+/comments/([a-z0-9]+)", page)))
found = get("posts/ids", ids=",".join(ids))  # cached by s1_global_2025


def source_url(p):
    if p.get("media_metadata"):
        order = [x["media_id"] for x in (p.get("gallery_data") or {}).get("items", [])] or list(p["media_metadata"])
        s = p["media_metadata"][order[0]].get("s", {})
        return s.get("u") or s.get("gif")
    if p.get("preview") and p["preview"].get("images"):
        return p["preview"]["images"][0]["source"]["url"]
    return None


ok = 0
for p in found:
    path = os.path.join(OUT, p["id"] + ".jpg")
    if os.path.exists(path):
        ok += 1
        continue
    url = source_url(p)
    if not url:
        print("no image:", p["id"], p["subreddit"], p.get("title", "")[:60])
        continue
    try:
        r = requests.get(htmllib.unescape(url), timeout=60,
                         headers={"User-Agent": "newcomer-alignment-study/0.1 (academic research)"})
        r.raise_for_status()
        im = Image.open(io.BytesIO(r.content)).convert("RGB")
        im.thumbnail((768, 768))
        if min(im.size) < 280:  # pad very wide/tall images up to Jev's minimum side
            canvas = Image.new("RGB", (max(im.width, 280), max(im.height, 280)), "white")
            canvas.paste(im, ((canvas.width - im.width) // 2, (canvas.height - im.height) // 2))
            im = canvas
        im.save(path, "JPEG", quality=85)
        ok += 1
    except Exception as e:  # noqa: BLE001
        print("failed:", p["id"], e)
print(ok, "of", len(found), "posts have an image")
