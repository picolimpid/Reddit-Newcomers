"""Shared paths and constants. The Reddit Newcomers Study 2 data is read (never written)
through the data/source symlink."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE = os.path.join(ROOT, "data", "source")
BENCH = os.path.join(ROOT, "data", "benchmark.parquet")
RESULTS = os.path.join(ROOT, "results")
DAY = 86400

# controls of the Reddit Newcomers models (REPORT §3.3): history size/activity + entry position
HIST = ["log_hist_n", "log_n_subs", "log_hist_len", "log_hist_rate", "hist_eng"]
ENTRY = ["log_delay", "top_level", "log_prior"]
CONTROLS = HIST + ENTRY

# pre-entry windows (days before the target post), as in REPORT §4.12 F4
WINDOWS = [(0, 30), (30, 90), (90, 365)]
