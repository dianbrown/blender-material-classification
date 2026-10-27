# v0.5 independent review

Fixed review base: `ca9efdf`. Initial reviewed head: `8be57b8`.
Specification: [release plan](RELEASE_V05_PLAN.md).

## Standards

The independent standards/correctness review found one P2: the installation
guide still pinned the executable hash from before the worker startup fix.
No documented-standard violations or actionable Fowler-baseline findings were
reported. The review checked installer path validation, input identity,
mutation checks, uninstall policy, native resources and worker startup.

## Spec

The independent specification review found the same P2: following the documented
command would reject the final executable before compilation. It also verified
the 21 sample paths, showcase hashes and infinite-loop GIF metadata. No P1
finding was reported. Actual full installation and publication were intentionally
excluded while those operations were pending.

## Resolution

Commit `98a76cc` corrects the documented executable identity. A fresh independent
verification confirmed that the command, package manifest and actual binary
match SHA-256 `34c5257c74fcc77ef37b3b6f3bfd0ea0e14ba4897bc6e178dd090deeb44805d2`
and 12,509,184 bytes. Both axes report no remaining P1/P2 in that verification
scope. The executable's Windows version remains 0.5.0 and the cube resources
are unchanged by the startup fix.

The full installer build initially refused 27 untracked Python bytecode files
created during a real smoke check. No installer was compiled from that dirty
inventory. Native worker startup now passes `-B`; its subprocess regression and
an actual packaged CPU prediction confirm that no bytecode files are created.
Only identified untracked cache files were removed; all inventoried payload
files were retained. The replacement binary and manifest were updated together
with the original identity retained in local QA evidence.

Summary: one resolved finding on each axis (the same stale documentation hash);
no unresolved scoped P1/P2. Build/install receipts and release checksums provide
the final operational evidence separately from these source reviews.
