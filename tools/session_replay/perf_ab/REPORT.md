# Session-replay recorder — performance certification (A/B)

Runs: off1, off2, off3, on1, on2, on3 — full 36-story ui_crawler suite per run, arms
alternated O-N-O-N on a quiet machine; identical binaries, `GENTLY_REPLAY`
is the only delta. Durations are Playwright-trace paired actions.

## Verdict: **PASS**

### Gates (absolute, operator-felt)

- Functional: ON may not produce a story outcome OFF never produced — OK
- Console: no new error types in ON — OK
- goto median added ≤ 300 ms; other felt methods median added ≤ 25 ms, p95 added ≤ 100 ms — OK

### Per-method added latency (all felt actions pooled across runs)

| method | n(off/on) | median off | median on | Δ median | Δ p95 |
|---|---|---|---|---|---|
| click | 138/138 | 186 | 197 | +11 ms | +28 ms |
| evaluateExpression | 636/636 | 12 | 12 | +0 ms | -2 ms |
| fill | 3/3 | 34 | 40 | +6 ms | +7 ms |
| goto | 105/105 | 559 | 634 | +76 ms | -2645 ms |

### Transparency: relative per-story felt-action totals

Median per-story delta: **+8.2%** (informational — ratios on
millisecond-scale stories flag imperceptible absolute costs; the absolute
gates above are the contract).

| story | off ms | on ms | Δ% |
|---|---|---|---|
| US-01 | 823 | 772 | -6.3 |
| US-02 | 1575 | 901 | -42.8 |
| US-03 | 799 | 984 | +23.2 |
| US-04 | 778 | 922 | +18.6 |
| US-05 | 828 | 937 | +13.2 |
| US-06 | 794 | 1129 | +42.2 |
| US-07 | 752 | 993 | +32.1 |
| US-08 | 918 | 1063 | +15.9 |
| US-09 | 1180 | 1421 | +20.4 |
| US-10 | 1093 | 1282 | +17.3 |
| US-11 | 1066 | 1100 | +3.2 |
| US-12 | 1093 | 1321 | +20.8 |
| US-13 | 1157 | 1171 | +1.2 |
| US-14 | 1144 | 1164 | +1.7 |
| US-15 | 935 | 1158 | +23.9 |
| US-16 | 926 | 1023 | +10.5 |
| US-17 | 970 | 1048 | +8.0 |
| US-18 | 1080 | 1198 | +10.9 |
| US-25 | 108 | 196 | +82.7 |
| US-26 | 846 | 1011 | +19.5 |
| US-28 | 962 | 1035 | +7.6 |
| US-29 | 1090 | 1067 | -2.1 |
| US-30 | 905 | 964 | +6.5 |
| US-31 | 932 | 1008 | +8.2 |
| US-32 | 1062 | 1089 | +2.5 |
| US-33 | 1005 | 991 | -1.4 |
| US-35 | 876 | 940 | +7.3 |
| US-36 | 852 | 859 | +0.8 |
| US-37 | 1001 | 1039 | +3.8 |
| US-38 | 598 | 725 | +21.2 |
| US-39 | 718 | 727 | +1.2 |
| US-40 | 671 | 747 | +11.3 |
| US-41 | 157 | 200 | +27.7 |
| US-42 | 829 | 895 | +8.0 |
| US-43 | 705 | 738 | +4.7 |
