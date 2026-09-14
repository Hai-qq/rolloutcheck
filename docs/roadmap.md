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

**P0 is partially validated**: local conversion feasibility, fail/pass controls
and small-model generation are established. The original large-model run and an
independent practical advantage are not established. P1's offline core is implemented. This public alpha
is an experimental artifact, not a declaration of full framework support.

## Next: verify usefulness in a framework workflow

1. Integrate with one actual engine/adapter boundary that exports raw IDs and
   stable turn identity. The current Transformers wrapper does not establish
   compatibility with slime/SGLang's scheduler, batching or stop semantics.
2. Obtain another authorized trajectory with a real debugging need, and compare
   the upstream logs/tests against trace extraction and portable evidence on
   that same case. Document which manual steps are actually removed.
3. Measure collector-on versus collector-off overhead on a fixed workload before
   making a performance claim. The current four-call timing record is not that
   benchmark. Verify CUDA/4070 execution separately if the target path requires it.
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
