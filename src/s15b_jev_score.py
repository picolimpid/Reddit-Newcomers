"""Stage 15b (Jev): model-based measurements with TypeSafe Jev (System One decision model).

PART=align    prior-interest alignment (proposal Method 2). One request per (user, target post):
              post title/body + post image, the user's 20 most-used prior subreddits and 40 most
              recent prior items. Pairs: newcomers x entry post, newcomers x placebo post (random
              target post from another subreddit), existing members x entry post.
              VARIANT=full (default) | subs (no item texts) | text (no subreddit names) | noimg
PART=comment  social signals in the first comment (newcomers + existing members): insider
              knowledge, outsider / member self-signalling, function, stance, hostility.
PART=reply    direct replies to first comments (newcomers + existing members): gatekeeping,
              welcoming, context-explaining, hostility, stance. Jev is not told who is a newcomer.
PART=post     target-post genre and how much insider knowledge it takes to get the post.
PART=window   pre-entry trajectory: community alignment from the subreddits a newcomer used 0-30,
              30-90 and 90-365 days before the post (names only, no image), cf. s14c F4.

LIMIT=n runs a pilot on the first n items (in a fixed random order) and prints the cost.
Output: data/<study>/jev_<part>[_<variant>].parquet. Run with STUDY=global."""
import base64
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

import jev
from api import DATA

PART = os.environ.get("PART", "align")
VARIANT = os.environ.get("VARIANT", "full")
LIMIT = int(os.environ.get("LIMIT", "0"))
WORKERS = int(os.environ.get("WORKERS", "8"))
N_ITEMS, N_SUBS, ITEM_CHARS = 40, 20, 200
BAD = {"[deleted]", "[removed]", ""}
BOTS = ("AutoModerator",)

posts = pd.read_parquet(os.path.join(DATA, "target_posts.parquet")).set_index("post_id")
at = pd.read_parquet(os.path.join(DATA, "analysis_table.parquet"))
IMG = os.path.join(DATA, "post_images")


def image_asset(pid):
    path = os.path.join(IMG, pid + ".jpg")
    if not os.path.exists(path):
        return None
    return {"type": "image", "format": "jpg", "data_base64": base64.b64encode(open(path, "rb").read()).decode()}


def post_view(pid, with_image=True):
    p = posts.loc[pid]
    view = {"community": "r/" + p.subreddit, "title": p.title}
    if p.selftext and p.selftext.strip() not in BAD:
        view["body"] = p.selftext.strip()[:1000]
    assets = None
    a = image_asset(pid) if with_image else None
    if a:
        view["image"] = "img-" + pid
        assets = {"img-" + pid: a}
    return view, assets


# ---------------------------------------------------------------- questions
LEVELS = ["No demonstrated alignment: nothing in the user's activity relates to it.",
          "Weak alignment: only a loose or incidental connection.",
          "Moderate alignment: the user has shown some interest in this.",
          "Strong alignment: the user clearly and repeatedly engages with this.",
          "Very strong and repeated alignment: this is one of the user's central interests."]
Q_ALIGN = {
    "align_post": {"type": "score", "criteria": LEVELS, "instructions":
                   "How strongly does `target_post` align with interests the user has demonstrated in their "
                   "prior Reddit activity (`user_subreddits`, `user_recent_activity`)?"},
    "align_community": {"type": "score", "criteria": LEVELS, "instructions":
                        "Ignoring this particular post: how strongly do the user's demonstrated interests align "
                        "with what the community in `target_post.community` is generally about (its usual topic "
                        "and culture)?"},
    "familiar": {"type": "noul", "instructions":
                 "Does the user's prior activity show that they already know or follow the topic, fandom or "
                 "culture of `target_post.community`, for example by discussing it, referencing it, or being "
                 "active in closely related communities?"},
}

FUNCTION = {
    "joke": "A joke, pun, meme reference or playful riff.",
    "reaction": "A short emotional reaction or exclamation (e.g. 'this is amazing', 'lol', 'wow').",
    "question": "Asks for information or clarification.",
    "information": "Provides facts, an explanation or expertise.",
    "personal": "Shares a personal experience or story.",
    "opinion": "States an opinion, judgment or argument.",
    "meta": "Talks about Reddit, the subreddit, the poster, or how the post was received.",
}
STANCE = {
    "agree": "Agrees with or supports the parent comment.",
    "disagree": "Disagrees with, corrects or challenges the parent comment.",
    "neutral": "Neither agrees nor disagrees: adds a joke, a tangent or new information.",
}


def shuffled(d, key):
    """Seeded per-item option order, so Jev's first-option preference averages out."""
    ks = list(d)
    random.Random(key).shuffle(ks)
    return {k: d[k] for k in ks}


def q_comment(key, is_reply):
    q = {
        "insider_knowledge": {"type": "noul", "instructions":
                              "Does `comment` show knowledge specific to `community` (its topic, in-jokes, jargon, "
                              "lore, recurring people or history) beyond what anyone could say from just seeing "
                              "this post?"},
        "outsider_signal": {"type": "noul", "instructions":
                            "Does the author of `comment` indicate that they are not a regular of `community` or "
                            "are unfamiliar with its topic (e.g. 'I don't follow this', 'why is this on my feed', "
                            "'not a fan but', asking what something is)?"},
        "member_signal": {"type": "noul", "instructions":
                          "Does the author of `comment` indicate that they are a fan, member or long-time follower "
                          "of `community` or its topic (e.g. 'as a long-time fan', 'we all know', calling the "
                          "community 'we' or 'us')?"},
        "hostile": {"type": "noul", "instructions": "Is `comment` hostile or insulting toward another person?"},
        "invites_response": {"type": "noul", "instructions":
                             "Does `comment` invite a response from others (asks something, challenges someone, "
                             "or directly addresses another user)?"},
        "function": {"type": "choice", "instructions": "What is the main function of `comment`?",
                     "criteria": shuffled(FUNCTION, key)},
    }
    if is_reply:
        q["stance"] = {"type": "choice", "instructions": "What is the stance of `comment` toward `parent_comment`?",
                       "criteria": shuffled(STANCE, key + "s")}
    return q


def q_reply(key):
    return {
        "gatekeeping": {"type": "noul", "instructions":
                        "Does `reply` treat the author of `first_comment` as an outsider or as not belonging in "
                        "`community`, e.g. mocking their lack of knowledge, saying they don't get it, calling them "
                        "a tourist, or telling them to go elsewhere or lurk more?"},
        "welcoming": {"type": "noul", "instructions":
                      "Is `reply` welcoming or helpful toward the author of `first_comment`, e.g. answering their "
                      "question, explaining context, inviting them in, or warmly agreeing?"},
        "explains_context": {"type": "noul", "instructions":
                             "Does `reply` explain background or community-specific context to the author of "
                             "`first_comment`?"},
        "hostile": {"type": "noul", "instructions":
                    "Is `reply` hostile or insulting toward the author of `first_comment`?"},
        "stance": {"type": "choice", "instructions": "What is the stance of `reply` toward `first_comment`?",
                   "criteria": shuffled(STANCE, key)},
    }


GENRE = {
    "meme_humor": "A meme, joke image or humorous screenshot.",
    "animals_wholesome": "Cute, funny or wholesome animals or people.",
    "impressive_satisfying": "An impressive skill, satisfying process or amazing sight.",
    "fail_karma": "A fail, accident, comeuppance or cringe moment.",
    "news_politics": "News, politics or a social issue.",
    "fandom_entertainment": "A TV show, film, game, anime, celebrity or other fandom.",
    "personal": "The poster's own appearance, life event, achievement, story or creation.",
    "other": "Anything else.",
}
Q_POST = {
    "genre": {"type": "choice", "instructions": "What kind of post is this?", "criteria": GENRE},
    "insider_needed": {"type": "score", "instructions":
                       "How much knowledge specific to this community or its topic is needed to understand or "
                       "appreciate this post?",
                       "criteria": ["None: anyone scrolling past would get it immediately.",
                                    "A little general background helps.",
                                    "Some familiarity with the topic is needed.",
                                    "Substantial knowledge of the topic or fandom is needed.",
                                    "Only insiders of the community would really understand it."]},
    "invites_discussion": {"type": "noul", "instructions":
                           "Does the post invite discussion (asks a question or raises a debatable issue) rather "
                           "than only being something to look at?"},
}


# ---------------------------------------------------------------- user history view
users = pd.read_parquet(os.path.join(DATA, "users.parquet"))
users = users[users["error"].isna()].set_index(["post_id", "author"])


WINDOWS = {"d0_30": (0, 30), "d30_90": (30, 90), "d90_365": (90, 365)}  # days before the post


def history_view(post_id, author, target_sub, window=None):
    hist = [h for h in json.loads(users.loc[(post_id, author), "history"])
            if h["sub"].lower() != target_sub.lower() and h["text"].strip() not in BAD
            and len(h["text"].strip()) >= 20]
    if window:
        T, (lo, hi) = posts.at[post_id, "created_utc"], WINDOWS[window]
        hist = [h for h in hist if T - hi * 86400 <= h["t"] < T - lo * 86400]
        if not hist:
            return None, None
    counts = pd.Series([h["sub"] for h in hist]).value_counts()
    subs = ", ".join(f"r/{s} ({n})" for s, n in counts.head(N_SUBS).items())
    recent = sorted(hist, key=lambda h: -h["t"])[:N_ITEMS]
    texts = [(f"r/{h['sub']}: " if VARIANT != "text" else "")
             + ("[post] " if h["kind"] == "p" else "") + " ".join(h["text"].split())[:ITEM_CHARS] for h in recent]
    return subs, texts


# ---------------------------------------------------------------- jobs
def align_jobs():
    rng = np.random.default_rng(0)
    pids = list(posts.index[posts.index.isin(at.post_id)])
    jobs = []
    for i, r in at[(at.newcomer_12m == 1) | (at.existing == 1)].iterrows():
        group = "newcomer" if r.newcomer_12m == 1 else "existing"
        targets = [("entry", r.post_id)]
        if group == "newcomer":
            others = [p for p in pids if posts.at[p, "subreddit"].lower() != r.subreddit.lower()]
            targets.append(("placebo", others[rng.integers(len(others))]))
        for role, pid in targets:
            jobs.append({"idx": i, "group": group, "role": role, "target_post": pid,
                         "post_id": r.post_id, "author": r.author, "subreddit": r.subreddit})
    return jobs


def window_jobs():
    """Pre-entry trajectory: subreddit-names-only community alignment in each window before the post."""
    return [{"idx": i, "group": "newcomer", "role": w, "target_post": r.post_id, "post_id": r.post_id,
             "author": r.author, "subreddit": r.subreddit}
            for i, r in at[at.newcomer_12m == 1].iterrows() for w in WINDOWS]


def run_window(j):
    subs, _ = history_view(j["post_id"], j["author"], j["subreddit"], window=j["role"])
    if subs is None:
        return {**j, "align_community": np.nan, "familiar": np.nan, "input_tokens": 0}
    view, assets = post_view(j["target_post"], with_image=False)  # the community, not the post, is the target
    out = jev.ask({"target_post": view, "user_subreddits": subs},
                  {k: Q_ALIGN[k] for k in ("align_community", "familiar")}, tag="align_window")
    a = out["answers"]
    return {**j, "align_community": a["align_community"]["score"], "familiar": a["familiar"]["noul"],
            "input_tokens": out["usage"]["input_tokens"]}


def run_align(j):
    view, assets = post_view(j["target_post"], with_image=VARIANT != "noimg")
    subs, texts = history_view(j["post_id"], j["author"], j["subreddit"])
    state = {"target_post": view}
    if VARIANT != "text":
        state["user_subreddits"] = subs
    if VARIANT != "subs":
        state["user_recent_activity"] = texts
    out = jev.ask(state, Q_ALIGN, assets=assets, tag=f"align_{VARIANT}")
    a = out["answers"]
    return {**j, "align_post": a["align_post"]["score"], "align_community": a["align_community"]["score"],
            "familiar": a["familiar"]["noul"], "align_post_conf": a["align_post"]["confidence"],
            "input_tokens": out["usage"]["input_tokens"]}


comments = pd.read_parquet(os.path.join(DATA, "thread_comments.parquet"))
by_id = comments.set_index("id")


def body_of(cid):
    return " ".join(str(by_id.at[cid, "body"]).split())[:1500] if cid in by_id.index else None


def comment_jobs():
    jobs = []
    for i, r in at[((at.newcomer_12m == 1) | (at.existing == 1)) & at.first_comment_id.notna()].iterrows():
        parent = by_id.at[r.first_comment_id, "parent_id"]
        jobs.append({"idx": i, "group": "newcomer" if r.newcomer_12m == 1 else "existing",
                     "post_id": r.post_id, "comment_id": r.first_comment_id,
                     "parent_id": parent[3:] if parent.startswith("t1_") else None})
    return jobs


def run_comment(j):
    p = posts.loc[j["post_id"]]
    state = {"community": "r/" + p.subreddit, "post_title": p.title}
    parent = body_of(j["parent_id"]) if j["parent_id"] else None
    if parent:
        state["parent_comment"] = parent
    state["comment"] = body_of(j["comment_id"])
    out = jev.ask(state, q_comment(j["comment_id"], bool(parent)), tag="comment")
    a = out["answers"]
    row = {**j, **{k: a[k]["noul"] for k in ("insider_knowledge", "outsider_signal", "member_signal",
                                             "hostile", "invites_response")},
           "function": a["function"]["choice"], "function_conf": a["function"]["confidence"],
           "input_tokens": out["usage"]["input_tokens"]}
    row |= {f"function_p_{k}": v for k, v in a["function"]["probabilities"].items()}
    if "stance" in a:
        row |= {"stance": a["stance"]["choice"], "p_disagree": a["stance"]["probabilities"]["disagree"]}
    return row


def reply_jobs():
    kids = comments[comments.parent_id.str.startswith("t1_")].copy()
    kids["parent"] = kids.parent_id.str[3:]
    jobs = []
    sel = at[((at.newcomer_12m == 1) | (at.existing == 1)) & at.first_comment_id.notna()]
    first = dict(zip(sel.first_comment_id, zip(sel.index, sel.author, sel.newcomer_12m, sel.post_id)))
    for k in kids[kids.parent.isin(first)].itertuples():
        i, author, nc, pid = first[k.parent]
        if k.author in (author, *BOTS) or str(k.body).strip() in BAD:
            continue
        jobs.append({"idx": i, "group": "newcomer" if nc == 1 else "existing", "post_id": pid,
                     "comment_id": k.parent, "reply_id": k.id, "replier": k.author})
    return jobs


def run_reply(j):
    p = posts.loc[j["post_id"]]
    state = {"community": "r/" + p.subreddit, "post_title": p.title,
             "first_comment": body_of(j["comment_id"]), "reply": body_of(j["reply_id"])}
    out = jev.ask(state, q_reply(j["reply_id"]), tag="reply")
    a = out["answers"]
    return {**j, **{k: a[k]["noul"] for k in ("gatekeeping", "welcoming", "explains_context", "hostile")},
            "stance": a["stance"]["choice"], "p_disagree": a["stance"]["probabilities"]["disagree"],
            "input_tokens": out["usage"]["input_tokens"]}


def post_jobs():
    return [{"post_id": p} for p in posts.index if p in set(at.post_id)]


def run_post(j):
    view, assets = post_view(j["post_id"])
    out = jev.ask(view, Q_POST, assets=assets, tag="post")
    a = out["answers"]
    return {**j, "genre": a["genre"]["choice"], "genre_conf": a["genre"]["confidence"],
            "insider_needed": a["insider_needed"]["score"], "invites_discussion": a["invites_discussion"]["noul"],
            "input_tokens": out["usage"]["input_tokens"]}


def _safe(run, j):
    try:
        return run(j)
    except jev.BudgetExceeded:
        raise
    except Exception as e:  # noqa: BLE001
        print("FAILED", j, str(e)[:300], flush=True)
        return None


if __name__ == "__main__":
    make, run = {"align": (align_jobs, run_align), "comment": (comment_jobs, run_comment),
                 "reply": (reply_jobs, run_reply), "post": (post_jobs, run_post),
                 "window": (window_jobs, run_window)}[PART]
    jobs = make()
    if PART == "align" and VARIANT in ("text", "noimg"):  # ablations: newcomer subsample only
        keep = set(np.random.default_rng(1).choice(sorted({j["idx"] for j in jobs if j["group"] == "newcomer"}),
                                                   600, replace=False))
        jobs = [j for j in jobs if j["idx"] in keep]
    if PART == "align" and VARIANT == "subs":
        jobs = [j for j in jobs if j["group"] == "newcomer"]
    random.Random(0).shuffle(jobs)
    if LIMIT:
        jobs = jobs[:LIMIT]
    tok0 = jev.spent()[0]
    print(f"{PART}/{VARIANT}: {len(jobs)} requests; spent so far ${jev.spent()[1]:.3f}")
    rows, failed = [], 0
    with ThreadPoolExecutor(WORKERS) as ex:
        for k, res in enumerate(ex.map(lambda j: _safe(run, j), jobs)):
            if res is None:
                failed += 1
            else:
                rows.append(res)
            if (k + 1) % 500 == 0:
                print(f"  {k + 1}/{len(jobs)}  spent ${jev.spent()[1]:.3f}", flush=True)
    df = pd.DataFrame(rows)
    tok = df.input_tokens.mean() if len(df) else float("nan")
    print(f"done: {len(df)} ok, {failed} failed; mean input tokens/request {tok:.0f}; "
          f"this run billed ${(jev.spent()[0] - tok0) * jev.USD_PER_INPUT_TOKEN:.3f}; "
          f"full-run estimate ${tok * len(make()) * jev.USD_PER_INPUT_TOKEN:.2f}; total spent ${jev.spent()[1]:.3f}")
    name = f"jev_{PART}" + (f"_{VARIANT}" if PART == "align" and VARIANT != "full" else "") + ("_pilot" if LIMIT else "")
    df.to_parquet(os.path.join(DATA, name + ".parquet"))
