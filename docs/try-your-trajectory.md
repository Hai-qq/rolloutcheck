# Try RolloutCheck on your own training handoff

The useful next validation is a developer investigating their own authorized
trajectory. Replaying this repository's fixture checks installation; it does not
count as independent adoption or a newly diagnosed case.

## First check: no GPU needed

From a fresh checkout, install the core and inspect a public fixture:

```sh
python -m venv .venv
# Linux/macOS:
.venv/bin/python -m pip install .
.venv/bin/rolloutcheck audit-samples \
  cases/observed/slime-training-native/short/results/default/handoff.json --format text
```

On Windows, use `.venv\Scripts\python.exe` and `.venv\Scripts\rolloutcheck.exe`.
Expected fixture result: matching contexts, 161 generated tokens, 16 trainable
positions and 145 unaccounted tokens. Adding `--require-all-generated` exits 1;
this is the explicitly requested retention requirement failing, not a crashed tool.

## Use your own case

1. Identify one session whose history or training-token retention surprised you.
   Record the expected policy and what your existing framework logs already show.
2. Save captured generation IDs and the actual emitted Sample tokens/masks using
   the [handoff example](training-handoff.md#audit-saved-handoff-data-offline).
   Do not combine sessions or invent missing masks/turn identities.
3. Run `rolloutcheck audit-samples your-handoff.json --format text`. Apply the
   all-generated requirement only when it reflects your actual intended policy.
4. If you prepare unpacked HF training batches, use the
   [batch boundary audit](training-job.md#catch-a-mask-losing-collator) after your
   actual collator. Investigate mismatches; do not automatically rewrite samples.
5. Record whether this located a concrete problem, explained intentional dropping,
   duplicated existing diagnostics, or could not apply. A negative result is useful.

## Feedback that would establish practical value

Use the repository's **Training workflow field report** issue form if you choose
to share. Include framework versions, intended policy, summary counts, what you
changed and whether the finding persisted on rerun. Describe manual steps or
measured time actually saved; leave them unknown if you did not measure them.

Raw token IDs can reveal prompt/output text. Start with aggregate counts and a
small authorized synthetic reproduction; sharing private training data is not
required. A private/local successful run is valid feedback without publishing its
raw traces. No script sends logs, telemetry or contact messages.

Track evidence separately: fixture reproduction, diagnosis on your own trajectory,
and incorporation into a recurring training regression. None implies the next.
There is currently no verified external field report.
