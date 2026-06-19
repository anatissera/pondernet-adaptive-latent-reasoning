#!/usr/bin/env python3
"""Read-only: classify every existing run dir into the new experiment layout.
Prints a TSV proposal; moves nothing. Run from repo root or pondernet/."""
import os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DIRS = ["outputs", "results", "models/checkpoints"]

# (regex on the prefix-stripped name) -> experiment dir
EXPERIMENTS = [
    (re.compile(r"^baseline-k\d+$|^fixedk-"), "01-simcot-baselines"),
    (re.compile(r"^lr\d|^warmstart-lr"),      "02-simcot-pondernet-early"),
    (re.compile(r"^gcfix-|^100k$"),           "03-simcot-pondernet-gcfix"),
    (re.compile(r"^gammasweep"),              "04-simcot-pondernet-gammasweep"),
    (re.compile(r"^k_recipe_sweep-|^k-recipe"), "05-simcot-pondernet-k-recipe"),
]
DEAD    = re.compile(r"joint|halthead")
SCRATCH = re.compile(r"^fixcheck|^fixedk-diagnostic|^_|console|^simcot-pondernet-default$|driver")

def strip_prefix(name):
    for p in ("simcot-pondernet-", "simcot-"):
        if name.startswith(p):
            return name[len(p):]
    return name

def classify(name):
    if SCRATCH.search(name):
        return "scratch", "", ""
    core = strip_prefix(name)
    if DEAD.search(core):
        return "dead", "", core
    for rx, exp in EXPERIMENTS:
        if rx.search(core):
            run = re.sub(r"^(gammasweep-|k_recipe_sweep-)", "", core)
            return "run", exp, run
    return "scratch", "", ""   # unknown → triage as scratch for manual review

def main():
    print("src\tkind\tklass\texperiment\trun\tdest")
    for d in DIRS:
        base = os.path.join(ROOT, d)
        if not os.path.isdir(base):
            continue
        for entry in sorted(os.listdir(base)):
            src = os.path.join(d, entry)
            kind = "dir" if os.path.isdir(os.path.join(base, entry)) else "file"
            klass, exp, run = classify(entry)
            if klass == "run":
                dest = f"{d}/{exp}/{run}"
            elif klass == "dead":
                dest = f"{d}/archive/{run or entry}"
            else:
                dest = f"{d}/archive/scratch/{entry}"
            print(f"{src}\t{kind}\t{klass}\t{exp}\t{run}\t{dest}")

if __name__ == "__main__":
    main()
