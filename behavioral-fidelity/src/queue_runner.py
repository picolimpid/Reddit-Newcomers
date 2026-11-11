"""Persistent GPU job queue (2026-11-06; v2 19:50). Jobs are lines in results/queue/jobs.txt:
    name | gpus | needs | shell command using {gpus} (and {util} for memory-aware jobs)
`gpus` is 'n' (n idle GPUs, < 1 GB used) or 'n:G' (n GPUs with >= G GiB free each; the command's {util} becomes
G / total, so a small model can share a GPU that another user's job half fills; best fit, so idle GPUs stay free
for big jobs). `needs` is '-' or a comma list of paths (relative to src/) that must exist first.
At most MAX_GPUS GPUs are held by this queue. Lines can be appended while it runs. Finished jobs are recorded in
results/queue/state.tsv (name, status, start, end, gpus); 'done' and 'failed(..)' are final. A job that dies at
startup because another process took the memory is recorded as 'busy' and retried (at most MAX_TRY times).
Running jobs are mirrored in running.tsv, so a restarted runner adopts them instead of starting duplicates
(status of an adopted job comes from its log: 'failed(?)' if it ends in a Traceback).
GPUs used by the user's own `vllm serve` processes (read from their CUDA_VISIBLE_DEVICES) and GPUs listed in
results/queue/AVOID (comma list) are never used, even when they have free memory.
Stop with: touch results/queue/STOP (running jobs are not killed).

    setsid nohup python3 -u queue_runner.py >> ../results/queue/runner.log 2>&1 < /dev/null &"""
import datetime as dt
import os
import subprocess
import time

Q = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "queue")
SRC = os.path.dirname(os.path.abspath(__file__))
ALLOWED = [int(x) for x in os.environ.get("ALLOWED", "4,5,6,7,8").split(",")]
MAX_GPUS = int(os.environ.get("MAX_GPUS", "3"))
MAX_TRY = 6
BUSY = ("less than desired GPU memory utilization", "CUDA out of memory", "No available memory for the cache blocks")


def now():
    return dt.datetime.now().strftime("%m-%d %H:%M")


def state():
    s, tries = {}, {}
    if os.path.exists(f"{Q}/state.tsv"):
        for line in open(f"{Q}/state.tsv"):
            p = line.rstrip("\n").split("\t")
            s[p[0]] = p[1]
            tries[p[0]] = tries.get(p[0], 0) + 1
    return s, tries


def jobs():
    out = []
    for line in open(f"{Q}/jobs.txt"):
        if line.strip() and not line.startswith("#"):
            name, g, needs, cmd = (x.strip() for x in line.split("|", 3))
            n, mem = (g.split(":") + [None])[:2]
            out.append((name, int(n), float(mem) if mem else None, [] if needs == "-" else needs.split(","), cmd))
    return out


def gpus():
    q = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True).stdout
    return {int(a): (int(b), int(c)) for a, b, c in (l.split(",") for l in q.strip().splitlines())}


def avoid():
    out = set()
    try:
        out |= {int(x) for x in open(f"{Q}/AVOID").read().replace(",", " ").split()}
    except (OSError, ValueError):
        pass
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ")
            if b"vllm serve" not in cmd:
                continue
            env = open(f"/proc/{pid}/environ", "rb").read().split(b"\0")
        except OSError:
            continue
        for e in env:
            if e.startswith(b"CUDA_VISIBLE_DEVICES="):
                out |= {int(x) for x in e.split(b"=", 1)[1].decode().split(",") if x.strip().isdigit()}
    return out


def tail(name, n=40):
    try:
        return "".join(open(f"{Q}/{name}.log", errors="replace").readlines()[-n:])
    except OSError:
        return ""


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def save_running(running):
    with open(f"{Q}/running.tsv", "w") as f:
        for name, (p, g, t0) in running.items():
            pid = p if isinstance(p, int) else p.pid
            f.write(f"{name}\t{pid}\t{','.join(map(str, g))}\t{t0}\n")


def adopt():
    running = {}
    if os.path.exists(f"{Q}/running.tsv"):
        for line in open(f"{Q}/running.tsv"):
            name, pid, g, t0 = line.rstrip("\n").split("\t")
            if alive(int(pid)):
                running[name] = (int(pid), [int(x) for x in g.split(",")], t0)
                print(f"{now()} adopted {name} pid {pid} on GPU {g}", flush=True)
    return running


def finished(p):
    if isinstance(p, int):
        return None if alive(p) else "adopted"
    return p.poll()


def main():
    running = adopt()
    save_running(running)
    while not os.path.exists(f"{Q}/STOP"):
        for name, (p, g, t0) in list(running.items()):
            rc = finished(p)
            if rc is None:
                continue
            log = tail(name)
            if rc == "adopted":
                st = "failed(?)" if "Traceback" in log else "done"
            elif rc == 0:
                st = "done"
            else:
                st = "busy" if any(b in log for b in BUSY) else f"failed({rc})"
            with open(f"{Q}/state.tsv", "a") as f:
                f.write(f"{name}\t{st}\t{t0}\t{now()}\t{','.join(map(str, g))}\n")
            print(f"{now()} {name} {st}", flush=True)
            del running[name]
            save_running(running)
        (s, tries), info = state(), gpus()
        held = {g for _, gs, _ in running.values() for g in gs}
        skip = avoid()
        for name, n, mem, needs, cmd in jobs():
            if name in running or s.get(name) == "done" or str(s.get(name, "")).startswith("failed"):
                continue
            if s.get(name) == "busy" and tries.get(name, 0) >= MAX_TRY:
                continue
            if not all(os.path.exists(os.path.join(SRC, x)) for x in needs):
                continue
            if len(held) + n > MAX_GPUS:
                continue  # a 1-GPU job further down may still fit (big jobs can wait; 11-08)
            free = {g: (t - u) / 1024 for g, (u, t) in info.items() if g in ALLOWED and g not in held and g not in skip}
            if mem is None:
                gs = [g for g in ALLOWED if g in free and info[g][0] < 1000][:n]
                util = None
            else:  # best fit: the fullest GPU that still leaves mem GiB (+1.2 GiB driver margin) free
                ok = sorted((f, g) for g, f in free.items() if f >= mem + 1.2)
                gs = [g for _, g in ok][:n]
                util = f"{int(mem / (min(info[g][1] for g in gs) / 1024) * 100) / 100:.2f}" if gs else None
            if len(gs) < n:
                continue  # a smaller or memory-aware job further down may still fit
            log = open(f"{Q}/{name}.log", "w")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(map(str, gs)), HF_HUB_OFFLINE="1",
                       HF_HOME="/data/hanqim2/.cache/huggingface")
            c = cmd.replace("{gpus}", ",".join(map(str, gs)))
            if util:
                c = c.replace("{util}", util)
            p = subprocess.Popen(c, shell=True, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=SRC)
            running[name] = (p, gs, now())
            save_running(running)
            held |= set(gs)
            print(f"{now()} start {name} on GPU {gs}" + (f" util {util}" if util else ""), flush=True)
            time.sleep(90)  # let vLLM allocate before the next check
            info = gpus()
        time.sleep(60)
    print(f"{now()} STOP file found; {len(running)} jobs still running (not killed)")


if __name__ == "__main__":
    main()
