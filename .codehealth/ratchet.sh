#!/usr/bin/env bash
# Code-health ratchet (WO-23, cce-code-health 0.3.0). The same file in every peclet repo.
#
#   .codehealth/ratchet.sh [BASE_REF]          (default origin/main; run `git fetch` first)
#
# Compares the working tree (committed and uncommitted changes, untracked files included) with the merge base of
# HEAD and BASE_REF through `codehealth diff`, over the touched in-scope files, and fails on
#   - a new brain function in any zone (user U, host H, device K),
#   - a new file smell in a file of zone U or H,
#   - a debt rise over the touched files in zone K (device kernels),
#   - an analysis error (the file would score 0 debt and hide the three checks above) or an invalid suppression.
# A debt rise in U or H alone passes (it is listed in the log). `codehealth diff` reads the config of the BASE:
# a base without [[zones]] cannot separate the zones, so the run says mode=unzoned and applies the tool's own
# stricter verdict instead (debt over all touched files, file smells in every zone).
# Prints one summary line; details in <git-dir>/codehealth-ratchet/ratchet.log. Exit 0 pass, 1 fail, 2 error.
set -u
repo=$(git rev-parse --show-toplevel 2>/dev/null) || { echo "RATCHET ERROR: not inside a git work tree"; exit 2; }
base_ref=${1:-origin/main}
ch=$HOME/.venvs/codehealth/bin/codehealth
[ -x "$ch" ] || ch=$(command -v codehealth) || { echo "RATCHET ERROR: codehealth not found"; exit 2; }
out=$(git -C "$repo" rev-parse --absolute-git-dir)/codehealth-ratchet
mkdir -p "$out" || exit 2
log=$out/ratchet.log
json=$out/diff.json
name=$(basename "$repo")
base=$(git -C "$repo" merge-base HEAD "$base_ref" 2>/dev/null) \
  || { echo "RATCHET ERROR $name: no merge base of HEAD and $base_ref"; exit 2; }
rm -f "$json"
"$ch" diff --repo "$repo" --base "$base" --json "$json" > "$log" 2>&1
rc=$?
if [ "$rc" -ge 2 ] || [ ! -s "$json" ]; then
  echo "RATCHET ERROR $name: codehealth diff exit $rc (log $log)"
  exit 2
fi

# The verdict, from the diff's JSON: details to the log (stdout), the summary line to stderr.
read -r -d '' judge <<'PY'
import json
import re
import sys

path, name, base, version, log = sys.argv[1:6]
d = json.load(open(path))
head, before = d["head"], d["base"]
zoned = "zones" in d
layer = {"user": "U", "host": "H", "device": "K"}
fails = []

base_brains = {f["name"] for r in before.values() for f in r["functions"] if f["brain"]}
brains = {"U": 0, "H": 0, "K": 0}
for p, r in sorted(head.items()):
    for f in r["functions"]:
        if f["brain"] and f["name"] not in base_brains:
            z = layer.get(f.get("zone"), "H")
            brains[z] += 1
            fails.append(f"new brain function in {z}: {p}:{f['line']} {f['name']} (nloc {f['nloc']})")

smells = {"U": 0, "H": 0, "K": 0}
other = []
for msg in d["failures"]:
    m = re.match(r"^(.*): new file smell\(s\) ", msg)
    if m:
        z = layer.get(head.get(m.group(1), {}).get("zone"), "H")
        smells[z] += 1
        if z != "K" or not zoned:
            fails.append(f"new file smell in {z}: {msg}")
        else:
            print(f"NOTE (K file smell, not a ratchet failure): {msg}")
    elif ": analysis error: " in msg or ": invalid suppression" in msg:
        fails.append(msg)
    elif msg.startswith("debt over touched files rose"):
        if not zoned:
            fails.append(f"unzoned base config, tool verdict: {msg}")
    elif ": new brain function " not in msg:
        other.append(msg)
for msg in other:
    print(f"NOTE (not a ratchet condition): {msg}")

if zoned:
    k = d["zones"].get("device", {"debt_before": 0.0, "debt_after": 0.0})
    kb, ka = k["debt_before"], k["debt_after"]
    if ka > kb + 1e-9:
        fails.append(f"zone K debt over the touched files rose {kb:.2f} -> {ka:.2f}")
    for z, x in d["zones"].items():
        print(f"zone {layer.get(z, z)} ({z}): debt {x['debt_before']:.2f} -> {x['debt_after']:.2f}, "
              f"new brains {x['new_brains']}")
    debt = f"K_debt={kb:.1f}->{ka:.1f}"
else:
    debt = f"debt={d['debt_base']:.1f}->{d['debt_head']:.1f}(all)"
for f in fails:
    print(f"FAIL {f}")
touched = len({*head, *before})
print(f"RATCHET {'FAIL' if fails else 'PASS'} {name} base={base[:10]} touched={touched} "
      f"new_brain_U/H/K={brains['U']}/{brains['H']}/{brains['K']} new_smell_U/H={smells['U']}/{smells['H']} "
      f"{debt} mode={'zoned' if zoned else 'unzoned'} other={len(other)} codehealth={version} log={log}",
      file=sys.stderr)
sys.exit(1 if fails else 0)
PY
version=$("$ch" --version 2>&1)
summary=$(python3 -c "$judge" "$json" "$name" "$base" "$version" "$log" 2>&1 >> "$log")
rc=$?
case "$summary" in
  "RATCHET "*) echo "$summary" | tee -a "$log"; exit "$rc" ;;
  *) echo "RATCHET ERROR $name: evaluation failed (log $log)"; echo "$summary" >> "$log"; exit 2 ;;
esac
