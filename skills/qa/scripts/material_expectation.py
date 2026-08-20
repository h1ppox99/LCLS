#!/usr/bin/env python3
"""Known-material expectation checker (QA method 09) — material-agnostic.

Judges an endpoint I(q) curve against first-principles expectations of a KNOWN
material. The registry (parsed from methods/09_known_material_expectation.md's
json block) carries EXPLICIT theoretical q values per reflection; assignment of
observed sharp peaks to reflections is done by D-VOTING — every (peak,
reflection) pairing implies a candidate sample-detector distance, and the
distance >= 2 peaks agree on wins. That agreement IS the identity confirmation
(distance-free: the vote is equivalent to matching tan-ratio sequences), so the
same code runs unchanged on LaB6, Si, CeO2, AgBh, Al2O3, ...

Checks:
  identity      — D-voting assignment; underdetermined (1 reachable ring) -> soft
  completeness  — every reachable reflection assigned (full valid range, no caps)
  anchors       — validated rings/anchors (when the registry has them): px + contrast
  geometry      — D_eff spread (model witness) and vs nominal (constant audit)
  extras        — sharp features predicted by neither layer -> parasitic candidates

Usage:
  python3 material_expectation.py --out-dir outputs/<run> --material LaB6 \
      [--nominal-dist-mm 190] [--report path.json]
  python3 material_expectation.py --selftest     # synthetic curves, all materials
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
METHOD_MD = HERE.parent / "methods" / "09_known_material_expectation.md"


def load_config() -> dict:
    md = METHOD_MD.read_text()
    return json.loads(re.search(r"```json\n(.*?)```", md, re.S).group(1))


# ---------------------------------------------------------------- geometry

def tan2t(q: float, lam: float) -> float:
    return float(np.tan(2 * np.arcsin(q * lam / (4 * np.pi))))


def r_of_q(q: float, dist_mm: float, lam: float, pix_mm: float) -> float:
    return dist_mm * tan2t(q, lam) / pix_mm


def q_of_r(r_px: float, dist_mm: float, lam: float, pix_mm: float) -> float:
    return float(4 * np.pi / lam * np.sin(np.arctan(r_px * pix_mm / dist_mm) / 2))


# ---------------------------------------------------------------- peak finding

def find_sharp_peaks(r, I, sigma_k: float, fwhm_max_px: float):
    """Prominence-fenced local maxima with FWHM; robust noise from MAD(dI)."""
    from scipy.signal import find_peaks, peak_widths
    ok = np.isfinite(I)
    noise = 1.4826 * np.median(np.abs(np.diff(I[ok]))) / np.sqrt(2)
    Ii = np.where(ok, I, np.interp(r, r[ok], I[ok]))
    idx, props = find_peaks(Ii, prominence=sigma_k * noise)
    widths = peak_widths(Ii, idx, rel_height=0.5)[0]
    peaks = [{"r_px": float(r[i]), "prominence": float(p), "fwhm_px": float(w)}
             for i, p, w in zip(idx, props["prominences"], widths)]
    return [p for p in peaks if p["fwhm_px"] <= fwhm_max_px], peaks, noise


def local_contrast(r, I, r0: float, half: int = 20, bg_off=(30, 70)):
    lo, hi = int(r0) - half, int(r0) + half
    win = I[max(0, lo):hi]
    if not np.isfinite(win).any():
        return None, None
    k = int(np.nanargmax(win))
    pk_r, pk_i = max(0, lo) + k, float(win[k])
    sb = []
    for s0, s1 in [(int(r0) - bg_off[1], int(r0) - bg_off[0]),
                   (int(r0) + bg_off[0], int(r0) + bg_off[1])]:
        seg = I[max(0, s0):s1]
        seg = seg[np.isfinite(seg)]
        if len(seg):
            sb.append(np.median(seg))
    bg = float(np.mean(sb)) if sb else np.nan
    contrast = (pk_i - bg) / bg if np.isfinite(bg) and bg > 0 else None
    return pk_r, contrast


# ---------------------------------------------------------------- D-voting

def assign_by_d_voting(sharp: list[dict], q_table: dict[str, float],
                       lam: float, pix: float, tol_pct: float,
                       d_bounds: tuple[float, float]):
    """Assign sharp peaks to reflections via distance voting.

    Every (peak, reflection) pairing implies D_cand = r*pix/tan(2theta_pred).
    Candidates are constrained to a broad plausibility window around the
    nominal distance (so junk peaks cannot elect an absurd geometry); within
    that window the vote stays distance-free. The D that the most peaks agree
    on (within tol_pct) wins — ties broken by smaller spread. >= 2 votes
    confirms material identity.
    Returns (assignment {hkl: peak}, D_mean, votes).
    """
    cands = []                       # (D_cand, peak_idx, hkl)
    for i, p in enumerate(sharp):
        for hkl, q in q_table.items():
            D = p["r_px"] * pix / tan2t(q, lam)
            if d_bounds[0] <= D <= d_bounds[1]:
                cands.append((D, i, hkl))
    best = (None, 0, None, np.inf)   # (D_mean, votes, cluster, spread)
    for D0, _, _ in cands:
        cluster = {}                 # peak_idx -> (D, hkl) closest per peak
        for D, i, hkl in cands:
            if abs(D - D0) / D0 * 100 <= tol_pct:
                if i not in cluster or abs(D - D0) < abs(cluster[i][0] - D0):
                    cluster[i] = (D, hkl)
        # one reflection may win only one peak: keep the closest peak per hkl
        by_hkl = {}
        for i, (D, hkl) in cluster.items():
            if hkl not in by_hkl or abs(D - D0) < abs(by_hkl[hkl][1] - D0):
                by_hkl[hkl] = (i, D)
        votes = len(by_hkl)
        ds = [d for _, d in by_hkl.values()]
        spread = (max(ds) - min(ds)) / np.mean(ds) if ds else np.inf
        if votes > best[1] or (votes == best[1] and spread < best[3]):
            best = (float(np.mean(ds)), votes, by_hkl, spread)
    if best[1] < 2:
        return {}, None, best[1]
    assignment = {hkl: sharp[i] for hkl, (i, _) in best[2].items()}
    return assignment, best[0], best[1]


# ---------------------------------------------------------------- main check

def run_check(out_dir: Path, material: str, nominal_dist_mm: float,
              lam: float | None = None, iq_path: Path | None = None) -> dict:
    cfg = load_config()
    lam = lam or cfg["lambda_A_default"]
    pix = cfg["pixel_mm"]
    if material not in cfg["materials"]:
        return {"material": material, "verdict": "skip",
                "reason": "material_not_in_registry"}
    mat = cfg["materials"][material]
    q_table = mat["q_hkl_invA"]

    arr = np.load(iq_path or (out_dir / "iq.npy"))
    r, I = arr["r_px"].astype(float), arr["I_mean"]
    valid = np.isfinite(I)
    r_max_valid = float(r[valid].max())

    escalations, notes = [], []
    sharp, all_peaks, noise = find_sharp_peaks(
        r, I, cfg["extra_feature_prominence_sigma"], cfg["sharp_fwhm_max_px"])

    # ---- identity via D-voting (generic, material-agnostic) ----------------
    lo, hi = cfg["d_vote_bounds_x_nominal"]
    assignment, D_vote, votes = assign_by_d_voting(
        sharp, q_table, lam, pix, cfg["ratio_tol_pct"],
        (nominal_dist_mm * lo, nominal_dist_mm * hi))
    identity = "confirmed" if votes >= 2 else "underdetermined"
    if votes >= 2:
        D_eff = D_vote
    else:
        # degenerate: <=1 reachable ring -> position-only fallback at nominal D
        D_eff = nominal_dist_mm
        escalations.append({"id": "identity_underdetermined", "severity": "soft",
                            "detail": f"only {votes} peak(s) vote — position-only "
                                      "fallback at nominal distance"})
        for hkl, q in sorted(q_table.items(), key=lambda t: t[1]):
            r_pred = r_of_q(q, nominal_dist_mm, lam, pix)
            if r_pred > r_max_valid:
                continue
            near = [p for p in sharp
                    if abs(p["r_px"] - r_pred) <= cfg["anchor_delta_px_max"]]
            if near:
                assignment[hkl] = near[0]

    q_max = q_of_r(r_max_valid, D_eff, lam, pix)
    margin = 1 - cfg["completeness_edge_margin_pct"] / 100

    # ---- per-ring table + D_eff stats --------------------------------------
    # validated px anchors are geometry-bound: gate on them only when this
    # run's geometry matches the geometry they were validated at
    val_deff = mat.get("validated_deff_mm")
    anchors_active = (val_deff is not None
                      and abs(D_eff - val_deff) / val_deff * 100
                      <= cfg["anchor_geometry_tol_pct"])
    if val_deff is not None and not anchors_active:
        notes.append(f"validated px anchors skipped: run D_eff {D_eff:.1f} mm vs "
                     f"validated geometry {val_deff} mm (> "
                     f"{cfg['anchor_geometry_tol_pct']}% apart)")
    rings_cfg = mat.get("lattice_rings_validated", {}) if anchors_active else {}
    lattice_rings, deffs = [], []
    for hkl, q in sorted(q_table.items(), key=lambda t: t[1]):
        if q > q_max:
            continue
        edge = q > q_max * margin
        p = assignment.get(hkl)
        entry = {"hkl": hkl, "q_pred_invA": q,
                 "r_pred_px": round(r_of_q(q, D_eff, lam, pix), 1),
                 "found": p is not None}
        if p is not None:
            pk_r, contrast = local_contrast(r, I, p["r_px"])
            deff = p["r_px"] * pix / tan2t(q, lam)
            entry.update({"r_obs_px": p["r_px"], "fwhm_px": round(p["fwhm_px"], 1),
                          "contrast": None if contrast is None else round(contrast, 4),
                          "D_eff_mm": round(deff, 2)})
            deffs.append((hkl, deff))
            spec = rings_cfg.get(hkl)
            if spec:
                if abs(p["r_px"] - spec["r_px"]) > cfg["anchor_delta_px_max"]:
                    escalations.append({"id": "validated_ring_moved", "severity": "hard",
                                        "detail": f"{hkl}: {p['r_px']:.0f} px vs "
                                                  f"validated {spec['r_px']} px"})
                if contrast is not None and contrast < spec["contrast_min"]:
                    escalations.append({"id": "lattice_ring_contrast_low",
                                        "severity": "hard",
                                        "detail": f"{hkl}: contrast {contrast:.3f} "
                                                  f"< {spec['contrast_min']}"})
        elif edge:
            notes.append(f"({hkl}) q={q:.3f} sits within "
                         f"{cfg['completeness_edge_margin_pct']}% of q_max — edge "
                         "case, not escalated")
        else:
            escalations.append({
                "id": "expected_lattice_ring_absent",
                "severity": "hard" if identity == "confirmed" else "soft",
                "detail": f"({hkl}) q={q:.3f} reachable "
                          f"(r~{r_of_q(q, D_eff, lam, pix):.0f} px) but not found"})
        lattice_rings.append(entry)

    # ratio test (reported for the first two assigned rings — equivalent to the vote)
    ratio_test = None
    found_rings = [e for e in lattice_rings if e["found"]]
    if len(found_rings) >= 2:
        a_, b_ = found_rings[0], found_rings[1]
        obs = b_["r_obs_px"] / a_["r_obs_px"]
        pred = tan2t(b_["q_pred_invA"], lam) / tan2t(a_["q_pred_invA"], lam)
        ratio_test = {"rings": [a_["hkl"], b_["hkl"]],
                      "observed": round(float(obs), 4),
                      "predicted": round(float(pred), 4),
                      "diff_pct": round((obs / pred - 1) * 100, 3)}

    deff_block = None
    if deffs and identity == "confirmed":
        vals = [d for _, d in deffs]
        mean = float(np.mean(vals))
        spread = (max(vals) - min(vals)) / mean * 100
        vs_nom = (mean / nominal_dist_mm - 1) * 100
        deff_block = {"per_ring": {h: round(d, 2) for h, d in deffs},
                      "mean_mm": round(mean, 2), "spread_pct": round(spread, 3),
                      "nominal_mm": nominal_dist_mm,
                      "vs_nominal_pct": round(vs_nom, 2)}
        if spread > cfg["deff_spread_pct_max"]:
            escalations.append({"id": "geometry_model_error", "severity": "soft",
                                "detail": f"D_eff spread {spread:.2f}% -> drift check"})
        if abs(vs_nom) > cfg["deff_vs_nominal_warn_pct"]:
            escalations.append({"id": "geometry_constant_mismatch", "severity": "soft",
                                "detail": f"D_eff {mean:.1f} mm vs nominal "
                                          f"{nominal_dist_mm} mm ({vs_nom:+.1f}%) — "
                                          "q-axis labels off"})
    notes.append(f"q_max (full valid range) = {q_max:.3f} 1/A at r = "
                 f"{r_max_valid:.0f} px — no radial cap applied")

    # ---- validated diffuse anchors (geometry-bound, same gate) -------------
    anchors = []
    anchor_cfg = mat.get("diffuse_anchors_validated", {}) if anchors_active else {}
    for r0, spec in anchor_cfg.items():
        pk_r, contrast = local_contrast(r, I, float(r0))
        found = pk_r is not None and abs(pk_r - float(r0)) <= cfg["anchor_delta_px_max"]
        ok = found and contrast is not None and contrast >= spec["contrast_min"]
        anchors.append({"r_px": float(r0), "found": bool(found), "r_obs_px": pk_r,
                        "contrast": None if contrast is None else round(contrast, 4),
                        "contrast_min": spec["contrast_min"], "pass": bool(ok)})
        if not ok:
            escalations.append({"id": "diffuse_anchor_failed", "severity": "soft",
                                "detail": f"anchor {r0} px: found={found}, "
                                          f"contrast={contrast}"})

    # ---- extras: sharp features predicted by neither layer -----------------
    known_r = ([e.get("r_obs_px") for e in lattice_rings if e["found"]]
               + [a["r_obs_px"] for a in anchors if a["found"]])
    extras = [p for p in sharp
              if all(abs(p["r_px"] - kr) > 15 for kr in known_r if kr is not None)]
    for p in extras:
        escalations.append({"id": "unattributed_sharp_feature", "severity": "soft",
                            "detail": f"sharp peak at r={p['r_px']:.0f} px "
                                      f"(fwhm {p['fwhm_px']:.0f} px) — parasitic "
                                      "candidate"})

    hard = [e for e in escalations if e["severity"] == "hard"]
    return {"material": material, "lambda_A": lam, "identity": identity,
            "votes": votes, "r_max_valid_px": r_max_valid,
            "q_max_invA": round(q_max, 3),
            "lattice_rings": lattice_rings, "ratio_test": ratio_test,
            "deff": deff_block, "diffuse_anchors": anchors,
            "extra_sharp_features": extras, "notes": notes,
            "escalations": escalations,
            "verdict": "fail" if hard else "pass"}


# ---------------------------------------------------------------- selftest

def selftest() -> int:
    """Synthetic curves per registry material: peaks at r(q, D_true) + background.

    Recovers D_true via D-voting with a deliberately wrong nominal distance —
    verifies identity, completeness, and the geometry_constant_mismatch flag.
    """
    cfg = load_config()
    lam, pix = cfg["lambda_A_default"], cfg["pixel_mm"]
    cases = {"LaB6": 100.0, "Si": 100.0, "CeO2": 100.0, "Al2O3": 100.0,
             "AgBh": 1000.0}
    rng = np.random.default_rng(4)
    ok_all = True
    for matname, D_true in cases.items():
        q_table = cfg["materials"][matname]["q_hkl_invA"]
        n = 1405
        r = np.arange(n, dtype=float)
        I = 300 * np.exp(-r / 800) + 100 + rng.normal(0, 5, n)
        n_rings = 0
        for q in q_table.values():
            rp = r_of_q(q, D_true, lam, pix)
            if rp < n - 30:
                I += 500 * np.exp(-0.5 * ((r - rp) / 3.0) ** 2)
                n_rings += 1
        arr = np.zeros(n, dtype=[("r_px", "f8"), ("I_mean", "f8")])
        arr["r_px"], arr["I_mean"] = r, I
        tmp = Path("/tmp") / f"_matexp_selftest_{matname}.npy"
        np.save(tmp, arr)
        rep = run_check(tmp.parent, matname, nominal_dist_mm=D_true * 0.95,
                        iq_path=tmp)
        d = rep.get("deff") or {}
        recovered = d.get("mean_mm")
        id_ok = (rep["identity"] == "confirmed") if n_rings >= 2 else \
                (rep["identity"] == "underdetermined")
        d_ok = recovered is None or abs(recovered - D_true) / D_true < 0.005
        gm = any(e["id"] == "geometry_constant_mismatch"
                 for e in rep["escalations"]) if n_rings >= 2 else True
        ok = rep["verdict"] == "pass" and id_ok and d_ok and gm
        ok_all &= ok
        print(f"  {matname:6s} D_true={D_true:6.0f} rings={n_rings} "
              f"identity={rep['identity']:15s} D_recovered={recovered} "
              f"verdict={rep['verdict']}  {'OK' if ok else 'FAIL'}")
        tmp.unlink()
    print("[selftest]", "ALL OK" if ok_all else "FAILURES above")
    return 0 if ok_all else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir")
    ap.add_argument("--material", default="LaB6")
    ap.add_argument("--nominal-dist-mm", type=float, default=190.0)
    ap.add_argument("--lambda-A", type=float, default=None)
    ap.add_argument("--report", default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if not args.out_dir:
        ap.error("--out-dir required unless --selftest")

    out_dir = Path(args.out_dir)
    rep = run_check(out_dir, args.material, args.nominal_dist_mm, args.lambda_A)
    dest = Path(args.report) if args.report else out_dir / "material_expectation_report.json"
    dest.write_text(json.dumps(rep, indent=2, default=float))

    print(f"[material_expectation] {args.material}: VERDICT {rep['verdict'].upper()} "
          f"(identity {rep.get('identity')}, {rep.get('votes')} votes)")
    if rep.get("ratio_test"):
        rt = rep["ratio_test"]
        print(f"  tan-ratio {rt['rings']}: obs {rt['observed']} vs pred "
              f"{rt['predicted']} ({rt['diff_pct']:+.2f}%)")
    if rep.get("deff"):
        d = rep["deff"]
        print(f"  D_eff {d['mean_mm']} mm (spread {d['spread_pct']}%), "
              f"nominal {d['nominal_mm']} mm ({d['vs_nominal_pct']:+.1f}%)")
    for e in rep.get("escalations", []):
        print(f"  [{e['severity']}] {e['id']}: {e['detail']}")
    for n_ in rep.get("notes", []):
        print(f"  note: {n_}")
    print(f"  report -> {dest}")
    return 0 if rep["verdict"] == "pass" else 5


if __name__ == "__main__":
    raise SystemExit(main())
