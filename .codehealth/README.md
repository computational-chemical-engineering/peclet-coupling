# Code-health ratchet (cce-code-health 0.3.0, same text in every peclet repo)

Files: `config.toml` (scoring profile `cpp@1` + `python@0`, scope, zones, `[docs]`; gate tiers where a refactor
campaign added them), `baseline.json` (scan of the commit the config landed on), `ratchet.sh` (the check).
Zones are the layers of the suite's code-health plan: `user` (U: Python, binding TUs, examples), `host` (H: setup,
launch sites, MPI plumbing), `device` (K: Kokkos kernels, found by their macros).

Before you commit a change to source in this repo, run from the repo's work tree:

    git fetch && .codehealth/ratchet.sh            # one summary line; exit 0 pass, 1 fail, 2 tool error

It compares your work tree (commits on your branch plus uncommitted and untracked files) with the merge base of
HEAD and origin/main, over the files you touched, and fails on: a new brain function in any zone; a new file
smell (large or huge file, duplication) in U or H; a debt rise over the touched files in K. A debt rise in U or H
alone passes. The log (`<git-dir>/codehealth-ratchet/ratchet.log`) names every failing file and function.

On a failure: fix it in the same change, within that change's own gates. Split the new brain function into named
phases (in a kernel: move the body whole, never reorder its arithmetic), or split the file that grew a smell. Do
not suppress, re-baseline or edit `config.toml` to pass: the check reads the BASE's config, so an edit here does
not loosen it anyway. If the fix would change numerics or a public name, stop and ask the repo's owner.
`mode=unzoned` means the base predates this config; the tool's stricter verdict is used until you rebase.

Baseline reproduction: `codehealth scan --repo . --ref <sha> --config .codehealth/config.toml --json out.json`
(byte-identical from a checkout of the same directory name; `meta.repo` holds that name).
