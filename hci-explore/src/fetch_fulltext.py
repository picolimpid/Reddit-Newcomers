"""Find papers on arXiv by exact-ish title, download PDF, convert to text (data/ext/papers/<slug>.txt)."""
import difflib, json, os, re, subprocess, sys, time, urllib.parse, urllib.request
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "ext", "papers")
UA = {"User-Agent": "hci-explore/0.1"}
def get(url, timeout=60):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout).read()
res = {}
for slug, title in json.load(open(sys.argv[1])).items():
    if os.path.exists(os.path.join(OUT, slug + ".txt")):
        print(slug, "cached"); continue
    head = re.split(r"[:?]", title)[0].strip()
    head = " ".join(head.split()[:8])
    xml = ""
    for q in (f'ti:"{head}"', " AND ".join([f"ti:{w}" for w in re.sub(r"[^A-Za-z0-9 ]", " ", title).split() if len(w) > 3][:6])):
        for _ in range(3):
            xml = get("https://export.arxiv.org/api/query?" + urllib.parse.urlencode({"search_query": q, "max_results": 5})).decode()
            if "<entry>" in xml: break
            time.sleep(5)
        if "<entry>" in xml: break
    best = None
    for e in re.findall(r"<entry>(.*?)</entry>", xml, re.S):
        t = re.sub(r"\s+", " ", re.search(r"<title>(.*?)</title>", e, re.S).group(1)).strip()
        aid = re.search(r"<id>http://arxiv.org/abs/(.*?)</id>", e).group(1)
        sim = difflib.SequenceMatcher(None, t.lower(), title.lower()).ratio()
        if not best or sim > best[0]:
            best = (sim, aid, t)
    if not best or best[0] < 0.8:
        res[slug] = {"status": "not found", "best": best}; print(slug, "NOT FOUND", best); time.sleep(3); continue
    pdf = os.path.join(OUT, slug + ".pdf")
    if not os.path.exists(pdf):
        open(pdf, "wb").write(get(f"https://arxiv.org/pdf/{best[1]}", 120))
    subprocess.run(["pdftotext", "-layout", pdf, pdf[:-4] + ".txt"], check=False)
    n = len(open(pdf[:-4] + ".txt", errors="ignore").read().split())
    res[slug] = {"status": "ok", "arxiv": best[1], "title": best[2], "words": n}
    print(slug, best[1], round(best[0], 2), n, "words"); time.sleep(3)
json.dump(res, open(os.path.join(OUT, "index.json"), "w"), indent=1)
