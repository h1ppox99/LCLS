# npy/ — deterministic XTC conversion (step 0 output)

- `frames_raw.npy` — (3201, 2, 512, 1024) uint16 **memmap** (`np.load(..., mmap_mode='r')`). Raw Jungfrau words: high 2 bits = gain mode (00 G0 / 01 G1 / 11 G2), low 14 bits = ADC. Calibrate with `(adc - calib/ped.npy[mode]) / calib/gain.npy[mode]` -> keV (see pipeline/xtclib.py).
- `shot_table.npz` — per-shot columns, global time order (same order as frames):
  fiducial, sec, nsec, stream, ipm2 (XPP-SB2-BMMON sum; offset NOT subtracted),
  xray (1 = beam, from EVR code 137/162), laser (1 = on, code 90/91),
  det_total (summed calibrated keV over good pixels),
  photon_px (# good pixels in 7-12 keV), nswitch (# non-G0 pixels, stuck-pixel proxy).
- ipm2 zero offset: estimate as median ipm2 over xray==0 shots.

## Extended monitor columns (step0b)

`step0b_extend_shot_table.py` merges every remaining per-shot scalar in the XTC
into `shot_table.npz` (same global time order; alignment verified bit-exact
against `ipm2`). Groups:

- `gasdet_f11/f12/f21/f22/f63/f64` — FEE gas detectors [mJ], upstream of all
  beamline optics. Decorrelated from the in-hutch monitors by spectral jitter
  (r ~= 0.69 vs ipm2): sanity anchors, not verdict monitors.
- `photon_energy_ev` — per-shot machine estimate of photon energy (BldDataEBeamV7).
  RELATIVE use only (jitter/drift): its absolute scale disagrees with the
  LaB6-ring-fitted geometry by ~+0.38% on Run0475. `ebeam_damage` != 0 marks
  shots where some machine fields are invalid (the always-invalid LTUpos/XTCAV
  fields make it nonzero on every Run0475 shot — judge per use, don't gate on it).
- `ebeam_charge_nc, ebeam_l3_mev, ebeam_pkcurr_bc1/bc2, ebeam_energy_bc1/bc2,
  ebeam_und_posx/posy/angx/angy, ebeam_dump_charge, ebeam_ltu250/ltu450` —
  accelerator diagnostics (weak direct value for normalization; kept for
  drift attribution studies).
- `pcav_t1_ps, pcav_t2_ps, pcav_q1_pc, pcav_q2_pc` — phase-cavity arrival-time
  diagnostics (pump-probe timing, not flux).
- `ipmfex22_*, ipmfex23_*, ipmfex26_*, ipmfex28_*` — four IPM/PIM diode boxes
  (IpmFex: 4 diode channels + sum + x/y position). Channel health on Run0475:
  `ipmfex22_ch1` dead (always 0); `ipmfex26_ch1` saturated (~1.21 const, poisons
  `ipmfex26_sum`); `ipmfex23` near saturation with x/ypos pinned at ~1.
  Healthy high-correlation channels: `ipmfex22_sum`, `ipmfex26_ch0+ch3`,
  `ipmfex28_sum` (all r ~= 0.90 vs det_total). Source identities per the
  pdsdata BldInfo enum (to confirm against the elog): 0x22 XppMonPim0,
  0x23 XppMonPim1, 0x26 XppSb3Pim, 0x28 XppEnds_Ipm0.
- `bmmon43_sum/xpos/ypos` — upstream (HX2-SB1) BMMON wave8: anchor only.
- `bmmon4b_xpos/ypos` — beam position at the ipm2 device (XPP-SB2 BMMON);
  `bmmon4b` sum IS the existing `ipm2` column (not duplicated).
- `bmmon4c_sum/xpos/ypos` — second XPP BMMON (likely SB3). Its xpos/ypos read
  identically on Run0475 — decode/device quirk, treat positions as opaque.

See the `skills/normalization/` monitor menu (README comparison table + one
file per reference parameter) for the evidence-based choice among these monitors.
