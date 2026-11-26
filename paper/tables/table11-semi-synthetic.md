# Table 11 — Semi-synthetic calibration

Real hourly bars (Yahoo Finance, 2025-10-22 to 2026-10-21, pre/post included) calibrate the return dynamics; freeze durations are measured Chainlink stale intervals. The full observation map is not independently calibrated by these inputs.

## Panel A. Return dynamics (regular session)

| Symbol | $n$ bars | $\sigma_h$ ($10^{-3}$) | $\sigma_h^{24/7}$ ($10^{-3}$) | $\hat\kappa_h$ ($10^{-3}$) | jump prob. ($\lvert r\rvert>4\sigma_h$) | jump sd / $\sigma_h$ |
| --- | --- | --- | --- | --- | --- | --- |
| SPY | 4147 | 2.21 | 1.47 | 0.90 | 0.0060 | 4.99 |
| QQQ | 4128 | 3.02 | 2.16 | 0.71 | 0.0060 | 4.18 |
| TSLA | 4128 | 6.51 | 4.86 | 2.01 | 0.0033 | 5.73 |
| NVDA | 4128 | 5.77 | 4.23 | 3.61 | 0.0047 | 5.08 |

## Panel B. Freeze duration and gap benchmarks

| Symbol | weekend $\tau$ median (h) | weekend $\tau$ p90 (h) | holiday $\tau$ median (h) | holiday $\tau$ p90 (h) | weekend gap / $\sigma_h\sqrt{\tau}$ | holiday gap / $\sigma_h\sqrt{\tau}$ | weekend gap / $g(\gamma=4)$ | holiday gap / $g(\gamma=4)$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SPY | 56.1 | 60.7 | 78.7 | 80.2 | 0.35 | 0.40 | 0.77 | 0.95 |
| QQQ | 55.4 | 58.3 | 76.3 | 87.5 | 0.40 | 0.38 | 0.86 | 0.90 |
| TSLA | 52.1 | 54.7 | 76.1 | 78.0 | 0.26 | 0.27 | 0.54 | 0.65 |
| NVDA | 52.1 | 55.7 | 76.0 | 77.6 | 0.28 | 0.29 | 0.59 | 0.70 |

## Panel C. Observation-map parameters (not independently calibrated here)

| Source | $\sigma_o$ | $\phi_{ext}$ | $\sigma_{ext}$ | $\phi_{stale}$ | $\sigma_{stale}$ |
| --- | --- | --- | --- | --- | --- |
| G0 true-$S$ diagnostic | 0.00050 | 0.805 | 0.00202 | 0.9784 | 0.00080 |
| observable-only estimate | 0.00041 | 0.774 | 0.00202 | 0.9802 | 0.00080 |

Notes. $\sigma_h$ is the standard deviation of hourly log returns within the regular session (09:30--16:00 ET, gaps $\le 90$ min); bars are stamped at the start of their interval, so the 16:00 bar is the first after-hours bar and is excluded. $\sigma_h^{24/7}=\sqrt{\sum r^2/\sum\Delta t}$ over all consecutive bars including overnight/weekend/holiday gaps, i.e. a per-elapsed-hour realised variance rate; $\hat\kappa_h=1-\hat\phi$ from an AR(1) on consecutive regular-session bars. Jump probability and conditional sd use the $4\sigma_h$ tail (5--9 events per symbol, so the conditional sd is noisy); the conditional sd is descriptive and not a jump-diffusion MLE. Overnight/weekend/holiday sd are log-close changes across duration buckets of 6--40 h, 40--70 h, and $>70$ h; the 40--70 h bucket can include midweek holiday closures, since it is not a calendar label. Freeze durations are measured Chainlink stale intervals for the matching 24/5 feed: weekend rows with gap $\ge 40$ h (28 gaps per feed), and the longest gap per labelled holiday (5 holidays per feed) before taking the median/p90 across holidays (short partial gaps inherit the calendar label and would otherwise bias the median down). The measured weekend stale interval (52--56 h) is shorter than the Friday-close-to-Monday-open market gap (about 65 h) because the 24/5 feed resumes on Sunday evening. In Panel B, gap/$\sigma_h\sqrt{\tau}$ is the continuous-diffusion benchmark, and gap/$g(\gamma=4)$ compares the observed total gap sd with the jump-only scale of the paper's stress variant ($g=\gamma\sigma_h\sqrt{1-e^{-\tau/40}}$, $\gamma=4$, $\tau_0=40$). The gap sd is measured across the market closure (about 65 h on weekends), while $\tau$ is the oracle stale interval (52--56 h); using the longer market duration in the denominator would lower the ratios, so the reported values are conservative. Values below 1 mean the observed gap is smaller than the benchmark. Because the observed gap also contains any freeze-period latent movement, the jump component itself is even smaller; the stress design is therefore conservative relative to these feeds. The observation map $h$ is not independently estimated by these calibration inputs: public bars provide return benchmarks and feed logs provide stale schedules, rather than a paired sample for fitting the full map; both synthetic observation-map estimates are reported, and the paper's coverage experiments use the observable-only one as their main specification.

Calibration summary. The simulator's DGP uses $\sigma_h=8.0\times10^{-3}$, $\kappa_h=3.0\times10^{-3}$, jump probability 0.005 with jump sd $3\sigma_h$, weekend freeze 65 h, and holiday freeze 81 h. The four feeds span $\sigma_h=2.21$--$6.51\times10^{-3}$, $\sigma_h^{24/7}=1.47$--$4.86\times10^{-3}$ (annualised 14--45%), $\kappa_h=0.71$--$3.61\times10^{-3}$, jump probability 0.0033--0.0060 (bracketing the simulator's 0.005) with conditional sd 4.2$\sigma_h$--5.7$\sigma_h$, weekend freeze 52.1--56.1 h, and holiday freeze 76.0--78.7 h. The simulator's $\sigma_h=8.0\times10^{-3}$ is above every session estimate and 1.6--5.4$\times$ the 24/7-equivalent rate (its annualised latent vol is about 75%); the environment is therefore a high-volatility stress setting, not a broad-market calibration. Its mean reversion sits inside the measured range ($0.71$--$3.61\times10^{-3}$), close to NVDA's upper end; its weekend freeze is longer than the measured 52--56 h; its 81 h holiday freeze is close to the measured 76--79 h; and its 3$\sigma_h$ jump sd is thinner-tailed than the measured 4.2$\sigma_h$--5.7$\sigma_h$ conditional sd.
