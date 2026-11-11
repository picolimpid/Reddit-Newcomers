"""Fetch candidate HCI papers (CHI, CSCW/PACM-HCI, UIST, IUI; 2024-2026) mentioning LLMs from OpenAlex.
Output: data/candidates.jsonl (title, venue, year, doi, abstract, cited_by_count)."""
import json, time, urllib.error, urllib.parse, urllib.request, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCES = {"S4363607743": "CHI", "S4210183893": "PACM-HCI(CSCW)", "S7407087746": "UIST", "S7407086775": "IUI"}
QUERY = 'LLM|"large language model"|"language models"'
def abstract(inv):
    if not inv: return ""
    pos = sorted((p, w) for w, ps in inv.items() for p in ps)
    return " ".join(w for _, w in pos)
out = open(os.path.join(ROOT, "data", "candidates.jsonl"), "w")
n = 0
for sid, venue in SOURCES.items():
    page = 1
    while page:
        flt = f"primary_location.source.id:{sid},publication_year:2024-2026,title_and_abstract.search:{QUERY}"
        url = ("https://api.openalex.org/works?" + urllib.parse.urlencode({
            "filter": flt, "per_page": 200, "page": page,
            "select": "id,doi,title,publication_year,cited_by_count,abstract_inverted_index,type"}))
        req = urllib.request.Request(url, headers={"User-Agent": "hci-explore/0.1"})
        for attempt in range(4):
            try:
                d = json.load(urllib.request.urlopen(req, timeout=60)); break
            except urllib.error.HTTPError as e:
                print(venue, page, e.code, e.read()[:300], flush=True); time.sleep(3 * (attempt + 1))
        else:
            break
        for w in d["results"]:
            ab = abstract(w.get("abstract_inverted_index"))
            out.write(json.dumps({"id": w["id"], "doi": w.get("doi"), "title": w["title"], "venue": venue,
                                  "year": w["publication_year"], "cites": w["cited_by_count"], "abstract": ab}) + "\n")
            n += 1
        page = page + 1 if len(d["results"]) == 200 else None
        time.sleep(0.2)
    print(venue, "cumulative", n, flush=True)
