# Feature engineering: WOE binning

## Build vs buy: benchmarking against optbinning

After implementing `WOEBinner` (`src/features/woe_binning.py`) from scratch and discussion with the supervisor was taken decision to include software engineering practice into the project - to check my implementation against a real binning library instead of assuming a hand implemented version is the right choise by default. Instead of rewriting the pipeline around a library, I ran a scoped benchmark: fit both `WOEBinner` and `optbinning.OptimalBinning` on the same three baseline features, train split only, and compared the resulting bins, WOE values, and information value (IV).

Script: `scripts/benchmark_binning.py`. For verification run it with:

```
python -m scripts.benchmark_binning
```

### Features compared

`dti`, `annual_inc`, `revol_util` - picked on purpose for different distributions and opposite monotonic directions (`dti`/`revol_util`: bad rate rises with the feature, `annual_inc`: bad rate falls)

### Results

| feature    | method         | bins | direction  | IV     |
| ---------- | -------------- | ---- | ---------- | ------ |
| dti        | WOEBinner      | 15   | increasing | 0.0775 |
| dti        | OptimalBinning | 15   | increasing | 0.0777 |
| annual_inc | WOEBinner      | 15   | decreasing | 0.0304 |
| annual_inc | OptimalBinning | 14   | decreasing | 0.0309 |
| revol_util | WOEBinner      | 15   | increasing | 0.0203 |
| revol_util | OptimalBinning | 13   | increasing | 0.0205 |

Across all three: same monotonic direction, IV within 0.3-1.6% relative of each other, bin counts differing by at most 2. Neither method had to fight the monotonicity constraint - the underlying bad rate relationship in this data is clean enough that a quantile and merge heuristic and a constraint solver converge on the same signal

### Where they actually differ

Bin edges are close but not identical, and the gap is methodological, not a correctness issue:

- `WOEBinner` starts from roughly 15 equal frequency quantile bins and only merges neighbors when a monotonicity violation or an undersized bin forces it to. For all three features here, none of the initial quantile bins violated monotonicity, so `WOEBinner`'s final bins are literally its starting quantile split - zero merges triggered
- `OptimalBinning` runs a CP-SAT solver that jointly picks cut points to directly maximize IV subject to the monotonicity and min bin size constraints, without being anchored to equal-frequency bins. That's why its bins are uneven in - size 56k to 168k rows within a single feature - and why it sometimes lands on fewer bins - it's willing to merge quantile regions where the marginal WOE difference between neighbors is small. For example, on `revol_util`'s upper tail `WOEBinner` keeps splitting `(56.2, inf]` into 7 separate quantile bins with only small WOE gaps between them, while `OptimalBinning` covers the same range in 5 wider bins.

### Manual implementation strong points

1. **Already tested and reviewed.** `tests/test_woe_binning.py` covers WOE computation, monotonicity enforcement, small bin merging, and missing value handling. Swapping to `optbinning` would mean re verifying a new dependency's behavior end to end for output that's within 1% of what I already have.
2. **Missing value handling is explicit by design, not a configuration option.** `WOEBinner` always gives missing values their own dedicated WOE bucket as part of the fit/transform contract. On `dti`, missing values score WOE = -0.80 - meaning a missing `dti` in this data correlates with _higher_ default risk, not just an absence of information.

3. **The audit log is the actual point.** Every bin merge logges a human readable reason in `overrides_`. For a project whose framing is regulatory defensibility of a PD model, being able to show and explain _why_ two bins got merged is worth more than a library call that hands back optimized cut points with no audit trail attached.
4. **The benchmark didn't show a case where the library changes the outcome.** If `OptimalBinning` had produced meaningfully higher IV, or caught a monotonicity violation my heuristic missed, that would be a real argument to switch. It didn't - the two methods agree closely on real features that the value of adopting the library would have been development speed, and that cost was already paid building `WOEBinner` in the first place.
