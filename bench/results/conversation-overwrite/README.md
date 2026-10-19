# Opt-in snapshot buffer overwrite growth

`STRATA_SNAPSHOT_NOFILL=1` skips value initialization when a snapshot capture
immediately overwrites the exposed range. Unset, `0`, and other values keep
zero-filled growth. This changes no cache policy or snapshot format.

## Why the copy covers the new storage

`conversation_kv_save` rejects a reusable prefix larger than the current buffer.
It then copies `[keep, size)` into the resized buffer. Since `keep` cannot exceed
the old size, this copy includes every newly exposed byte. A failed capture
must be discarded; it cannot publish partially written storage. The buffer
test checks allocation failure, callback failure, discard and retry, retained
prefixes, segment boundaries, shrink/regrow, and copy/move behavior.

The raw segment storage also backs ordinary `resize`. Ordinary growth still
fills the new bytes, including bytes reused after a shrink. The opt-in switch
does not restore `std::vector<uint8_t>` as the segment implementation.

## Historical engine evidence, not a backport result

`historical-v0145.csv` contains numeric fields extracted from existing local
engine logs. Source SHA-256 values identify the full original logs, which are
not published because they contain machine-specific paths. No model, pack,
prompt, config, credentials, or binary is included.

The historical workload used an experimental v0.1.45 CUDA binary on Windows,
two RTX 3090 24 GB cards, Flash-Next IQ4_XS, a 262,144-token context, an 8 GiB
conversation cache, four slots, a 65,536-token parking minimum, and fp16 KV.
It opened a roughly 245K-token synthetic document, sent a short unrelated
request to force parking, then returned to the long document three times.
Both settings used the same patched binary. The measured field is the engine's
park duration, not the whole request. Zero reused bytes identifies cold parks.

All six available named-run logs are included, including the slower overwrite
run recorded during other host work. A seventh run in the earlier report
could not be mapped to a standalone log and is omitted. This is a small,
historical, non-randomized sample; it motivates further testing rather than a
claim about all workloads or current main. Do not combine it with new runs.

## Validation status

- v0.1.44.1 backport: CUDA sm_86 build succeeds; buffer test 92 checks,
  conversation cache test 4,219 checks; split failure injection 36 cases pass.
- Current-main candidate: standalone buffer test 92 checks passes with MSVC
  19.44, C++17, optimized build. Full current-main CUDA build is not yet run.
- Same-binary v0.1.44.1 256K A/B, prefix edits, alternating long histories, HTTP
  error recovery, and a NOFILL-enabled 3-hour/50-request soak are pending.
- HIP and SYCL builds are not run; their toolchains are unavailable here.
- Keep this PR as a draft until the required backend and live checks finish.

## Reproduce buffer validation

Configure with `-DSTRATA_BUILD_CONVERSATION_TESTS=ON`, build
`conversation_buffer_test`, then run `ctest -R conversation_buffer_test
--output-on-failure` from the build directory. The test is host-only.

For an engine comparison, build once and record its SHA-256. Start a fresh
process for each setting, alternating `0,1` at least five times. Keep the
model, cache, GPU split, prompts, and generation parameters fixed. Record cold
and retained-prefix parks separately, request latency, all HTTP failures,
retrieval answers, token counts, evictions and memory. Check that cold opens
have zero cached tokens and actual park events occur before reporting times.
Compare outputs; do not assume temperature zero guarantees determinism.
