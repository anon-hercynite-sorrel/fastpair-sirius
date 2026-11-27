# SPDX-License-Identifier: Apache-2.0
"""Run inside GDB. Observe the real matchers without instrumenting the residual evaluator."""

from collections import Counter
import os

import gdb


class Call(gdb.Breakpoint):
    def __init__(self, symbol, label):
        super().__init__(symbol, internal=True)
        self.label = label

    def stop(self):
        if active:
            observed.add(self.label)
        return False


class Checkpoint(gdb.Breakpoint):
    def stop(self):
        global active
        expected = gdb.parse_and_eval("(char*)sirius_test_expected_matcher")
        if int(expected):
            if any(probe.pending for probe in probes):
                failures.append(("unresolved entry points", [p.label for p in probes if p.pending]))
            active = expected.string()
            observed.clear()
        else:
            # Projection may gather. COUNT(*) must never enter the prefix-scan,
            # index, or gather entry points, and must launch its direct popcount.
            if active == "count_only":
                wanted = {"swar", "count"}
                actual = observed
            elif active == "count_off":
                wanted = set()
                actual = observed
            else:
                wanted = {active}
                actual = observed & matchers
            print(f"decoded LIKE trace: {active=} observed={sorted(observed)}")
            if actual != wanted:
                failures.append((active, sorted(actual), sorted(wanted)))
            completed[active] += 1
            active = None
        return False


active = None
completed = Counter()
observed = set()
failures = []
symbols = {
    "contains": "cudf::strings::contains",
    "prefix": "cudf::strings::starts_with",
    "suffix": "cudf::strings::ends_with",
    "cudf_like": "cudf::strings::like",
    "swar": "sirius::like_multiliteral",
}
matchers = set(symbols)
symbols.update(
    count="sirius::codegen::count_masked_rows",
    prefix_scan="sirius::codegen::run_selection_cnt",
    indices="sirius::codegen::mask_to_row_indices",
    gather="cudf::gather",
)
gdb.execute("set pagination off")
gdb.execute("set disable-randomization off")
gdb.execute("set breakpoint pending on")
probes = [Call(symbol, label) for label, symbol in symbols.items()]
Checkpoint("sirius_test_matcher_checkpoint", internal=True)
gdb.execute("run")

if os.environ["SIRIUS_TEST_LIKE_TRACE_SCOPE"] == "count":
    mode = "count_only" if os.environ["SIRIUS_EXP_DECODED_LIKE"] == "1" else "count_off"
    required = Counter({mode: 1})
else:
    required = Counter(contains=4, prefix=2, suffix=2, swar=2, cudf_like=2)
if failures or completed != required or gdb.parse_and_eval("$_exitcode") != 0:
    print(f"decoded LIKE trace failed: {failures=} {completed=} {required=}")
    gdb.execute("quit 1")
gdb.execute("quit 0")
