# Third-party code and assets

RolloutCheck's original code is MIT licensed. This does not relicense third-party code.

`integrations/slime/upstream_helpers.py` contains a verbatim excerpt of
`slime/agent/adapters/common.py` from THUDM/slime commit
`885a09e852e5272d497936370707cc604830f805`, under Apache-2.0.
See [the retained upstream license](integrations/slime/slime-LICENSE) and
[source digests](integrations/slime/source.json).
Only a standard-library import header and provenance comments have been added.
The excerpt runs as local Python; the full adapter and its dependencies are not imported.

The reasoning-preservation and canonical-continuation fixes belong to upstream
PR [#2287](https://github.com/THUDM/slime/pull/2287), authored by `zy20031230`.
They are comparison controls, not RolloutCheck contributions.
The motivating issue is [#2288](https://github.com/THUDM/slime/issues/2288).
PR status was open and unmerged when checked on 2026-09-14.

The optional experiment downloads Qwen/Qwen3-0.6B tokenizer/configuration files
at revision `c1899de289a04d12100db370d81485cdf75e47ca`, plus the model repository's
Apache-2.0 LICENSE. Asset files are not bundled in this Git repository or wheel.
The explicit preparation command verifies each file against
`integrations/slime/assets.lock.json`. It never downloads model weights.

The experimental messages are hand-authored public fixtures. They do not contain
the original issue author's rollout data, and no upstream performance numbers
are presented as RolloutCheck measurements.
