# Roadmap and decision gates

The goal is a useful local workflow for engineers maintaining multi-turn rollout
conversions. Features follow demonstrated debugging needs.

## Current prototype

Implemented: bounded JSON loading, explicit applicability/continuity checks,
history-prefix comparison, captured-versus-derived boundary reporting,
witness-only evidence export, and one repeatable upstream-helper conversion
experiment with an existing upstream assertion as the detection baseline.
The v0.1.0a2 increment adds a single-sequence Transformers collector, bounded JSONL
trace inspection and extraction, and a real Qwen3-0.6B MPS generation contrast
with explicit EOS accounting. See [the observed case](observed-case.md).
The v0.1.0a3 increment adds slime debug-callback collection, persistent capture-gap
reporting and a served-turn count check. The unmodified slime HTTP adapter has
been tested with scripted controls and actual local Qwen generation through a
Transformers service. See [the integration](slime-integration.md).
The v0.1.0a4 increment adds bounded native-response validation and an opt-in
SGLang runner. A real RTX 4070 SUPER / WSL2 run captures six requests, verifies
wire/callback ID equality and records stop-token retention. See
[the native evidence and limits](native-sglang.md).
The v0.1.0a5 increment adds complete trace evidence bundles and offline bundle
verification. A bundle keeps the original bytes and capture gaps, and its report
is recomputed on verification. This removes manual source/report packaging from
the saved-trace workflow; independent developer benefit still needs evaluation.
The v0.1.0a6 increment closes a collector-lifecycle gap: new v2 traces require an
explicit end record with an independently declared count and a digest of preceding
bytes. Interrupted or unchecked traces cannot aggregate to PASS. The real slime
HTTP adapter now has six controlled lifecycle scenarios, including omitted health
checks. Legacy v1 evidence is preserved. See [the regression and migration](trace-completion.md).

**P0 is partially validated**: local conversion feasibility, fail/pass controls
and small-model generation are established. The original large-model run and an
independent practical advantage are not established. P1's offline core is implemented. This public alpha
is an experimental artifact, not a declaration of full framework support.

## Next: verify usefulness in a framework workflow

The next maturity gates are evidence-based: a usable controller integration,
measured overhead on its actual workload, and another developer reproducing or
using a diagnostic on their own authorized case. Passing the package tests or
adding a version does not close these gates.

1. Extend beyond the verified sequential SGLang 0.5.9 / Qwen3-0.6B configuration
   only when a concrete workflow requires it. Concurrent ancestry and other
   adapters/backends remain unverified; GPU access is now available.
2. Obtain another authorized trajectory with a real debugging need, and compare
   the upstream logs/tests against trace extraction and portable evidence on
   that same case. Document which manual steps are actually removed.
3. Measure collector-on versus collector-off overhead on a fixed workload before
   making a performance claim. The current four-call timing record is not that
   benchmark. The native run establishes CUDA execution, not collector overhead.
4. Add conversion replay or shrinking only when a captured case needs it;
   inspecting saved IDs is explicitly witness-only.

## Then: independent use

Seek voluntary feedback from relevant maintainers/users after a clean-environment
reproduction works. Separate independent reproduction, use on another developer's
own case, and inclusion in a recurring regression process. Stars, downloads, and
brief acknowledgments are not proxies for those outcomes.

The project can remain a small diagnostic utility if that is what proves useful.
If no practical increment is demonstrated, revisit the scope before expanding it.

## Conditional: shrinking

Only add shrinking when an authentic case has meaningful removable structure and
the user needs a smaller executable reproduction. Start with whole interactions
and a fixed target-failure predicate. Compare with the same validity checks,
budget, and existing message-level ddmin baseline. Never count broken inputs or
timeouts as preservation of the target failure. Do not claim global minimality.

Deferred: a second framework integration, generic plugin execution, dashboards,
real-time alerts, automatic repair, full training correctness checks, and training
performance claims. A rented server is an optional resource for a specific future
experiment, not a substitute for validating the product need.
