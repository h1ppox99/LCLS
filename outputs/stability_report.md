# Stability report — 3 trials (no-llm baseline)

## Decisions
- fields compared (rationales excluded): 9
- all decision fields identical across trials ✓

## Accumulation
- kept shots per trial: [1781, 1781, 1781] ✓ identical
- sum_total_keV spread: 0.0000% ✓

## Mask
- masked pixels per trial: [131856, 131856, 131856]
- IoU(trial1, trial2) = 1.0000 ✓
- IoU(trial1, trial3) = 1.0000 ✓
- IoU(trial2, trial3) = 1.0000 ✓

## Masked sum (endpoint)
- max relative diff on shared pixels (t1,t2): 0.000e+00
- max relative diff on shared pixels (t1,t3): 0.000e+00
- max relative diff on shared pixels (t2,t3): 0.000e+00

## Verdict: **STABLE**
