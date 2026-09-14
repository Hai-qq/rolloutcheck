# Roadmap and decision gates

The goal is a useful local workflow for engineers maintaining multi-turn rollout
conversions. Features follow demonstrated debugging needs.

## Current prototype

Implemented: bounded JSON loading, explicit applicability/continuity checks,
history-prefix comparison, captured-versus-derived boundary reporting,
witness-only evidence export, and one repeatable upstream-helper conversion
experiment with an existing upstream assertion as the detection baseline.

**P0 is partially validated**: local conversion feasibility and fail/pass controls
are established. The original large-model run and an independent practical
advantage are not established. P1's offline core is implemented. This public alpha
is an experimental artifact, not a declaration of full framework support.

## Next: real collection and a portable regression workflow

1. Identify one supported engine/adapter boundary that exports all required raw
   IDs and stable turn identity without manually reconstructing messages.
2. Capture a real small-model trajectory with explicit stop/EOS semantics.
   Record OS, GPU/VRAM, runtime versions, and collection overhead before claiming
   compatibility. CPU-controlled cases remain a separate evidence category.
3. Build a thin, opt-in collector for that verified path; establish before/after
   snapshots and an independently reproducible regression artifact.
4. Compare the actual upstream logs/tests and the new workflow on the same case.
   Find out which manual steps remain and whether RolloutCheck removes any.

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
