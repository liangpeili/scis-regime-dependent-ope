# Table 5 — Observable-only observation map $h$ (L3)

## Panel A. Point estimates

| Specification | $\\hat h_{\\rm open,sd}$ | $\\hat h_{\\rm ext,\\phi}$ | $\\hat h_{\\rm ext,sd}$ | $\\hat h_{\\rm stale,\\phi}$ | $\\hat h_{\\rm stale,sd}$ |
| --- | --- | --- | --- | --- | --- |
| truth | 0.000500 | 0.8000 | 0.002000 | 0.9800 | 0.000800 |
| G0 (true $S$ diagnostic) | 0.000496 | 0.8049 | 0.002016 | 0.9784 | 0.000797 |
| observable, z=3 (main) | 0.000410 | 0.7741 | 0.002023 | 0.9802 | 0.000798 |
| observable, z=2.5 | 0.000304 | 0.8455 | 0.001696 | 0.9814 | 0.000781 |
| observable, z=4 | 0.000469 | 0.7481 | 0.002116 | 0.9797 | 0.000802 |
| observable, $\sigma$ from OPEN | 0.000380 | 0.7466 | 0.002173 | 0.9801 | 0.000798 |
| unmasked MLE | 0.000467 | 0.7481 | 0.002099 | 0.9796 | 0.000802 |

Notes. Truth: $h_{\rm open,sd}=0.0005$, $h_{\rm ext,\phi}=0.8$,
$h_{\rm ext,sd}=0.002$, $h_{\rm stale,\phi}=0.98$, $h_{\rm stale,sd}=0.0008$.
The G0 row uses true $S$ as a diagnostic and is not an observable estimator.
`open_sd` and `ext_phi` are weakly identified from $P$ alone; the trimming
threshold z moves `ext_phi` from 0.7481 (z=4)
through 0.7741 (z=3) to 0.8455 (z=2.5).

## Panel B. Uncertainty for the weakly identified parameters

| Parameter | Bootstrap SE | Hessian SE | 90% bootstrap interval |
| --- | ---: | ---: | --- |
| $h_{\rm ext,\phi}$ | 0.0142 | 0.0176 | [0.7480, 0.7930] |
| $h_{\rm open,sd}$ | 0.000059 | 0.000056 | [0.000316, 0.000505] |

The 90% bootstrap interval for $h_{\rm ext,\phi}$ is
[0.7480, 0.7930], whose upper end lies
below the simulator truth 0.8. Because the trimming threshold is a researcher
choice, this interval is specification-specific, not a uniform confidence
statement.

## Panel C. Coverage sensitivity to the $h$ specification

| Policy | Specification | $G$ within | $G_r$ within |
| --- | --- | --- | --- |
| Wide | truth (G0) | 0.9297 | 0.9062 |
| Wide | observable $h$ (z=3) | 0.9258 | 0.9059 |
| Wide | observable $h$, z=2.5 | 0.9277 | 0.9059 |
| Wide | observable $h$, z=4 | 0.9238 | 0.9055 |
| Wide | observable $h$, $\sigma$ from OPEN | 0.9238 | 0.9059 |
| Narrow | truth (G0) | 0.9199 | 0.9059 |
| Narrow | observable $h$ (z=3) | 0.9238 | 0.9062 |
| Narrow | observable $h$, z=2.5 | 0.9219 | 0.9035 |
| Narrow | observable $h$, z=4 | 0.9199 | 0.9062 |
| Narrow | observable $h$, $\sigma$ from OPEN | 0.9238 | 0.9043 |

Identified raw: the pooled $G$ spread is 0.6 pp (Wide) and
0.4 pp (Narrow); the $G_r$ spread is 0.1 pp and
0.3 pp. The largest per-regime five-specification spread is
1.6 pp (identified raw, Narrow,
EXTENDED). Across all conformal score/pipeline combinations the largest
per-regime spread is 2.7 pp (identified
studentized, Wide, OPEN); per-regime cells are noisy,
and the pooled estimands are the stable objects.

Notes. Point estimates only; nominal coverage is 0.90 ($\alpha=0.1$). No 95%
interval is attached to the individual coverage cells; the uncertainty object
is the across-specification spread. The five specifications differ only in the
$h$ input of the same trajectory pipeline, so they are a specification curve,
not independent replications. The `truth (G0)` rows in Panel C use the
true-$S$ diagnostic $h$, not an observable estimator; the point estimate of $h$
is not used as a certificate.
