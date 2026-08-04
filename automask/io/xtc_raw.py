#!/usr/bin/env python3
"""
io/xtc_raw.py -- read the per-shot SCALARS out of an XTC stream without psana.

`io/read_xtc.py` is the real reader and stays the one to use wherever psana is
available. This module exists because the identification study
(`studies/loss_identification.py`) has five claims that need real data, and on a
checkout where psana cannot be installed -- it ships as a linux-64/osx-64 conda
package and this is osx-arm64 -- every one of them was blocked on the
environment rather than on the physics.

XTC is a container format, and the parts holding the per-shot scalars are fixed
little-endian C structs. So the scalars are readable with `struct`. The frames
are not, in any useful sense: a Jungfrau payload is raw 14-bit ADC plus a 2-bit
gain-stage tag, and turning that into the calibrated array the pipeline consumes
means reproducing psana's pedestal/gain/common-mode chain and its geometry
parsing. This module deliberately stops at the scalars.

WHAT IS DECODED, AND HOW EACH DECODE WAS CHECKED. None of the layouts below are
taken on faith; each has an independent consistency check, because a struct
misread produces plausible-looking numbers rather than an error.

    EVR event codes    `Id_EvrData` v4: uint32 count, then 12-byte FIFO events
                       (timestampHigh, timestampLow, eventCode).
                       CHECK: code 137 -- documented in DATA.md as 'Beam On' --
                       is present on 98% of events, and the codes seen are
                       exactly the stock XPP set (30-34, 40-44, 90/91, 137,
                       140-144).
    analog input       `BldDataAnalogInputV1`: uint32 nchannels, then that many
                       float64 volts. CHECK: nchannels reads 16, and channels
                       2 and 3 sit at either ~0 or ~5.05 V, which is exactly
                       what DATA.md documents for the CC and VCC shutters.
    IPM / diode fex    `Lusi.IpmFexV1`: float32 channel[4], sum, xpos, ypos.
                       CHECK: `sum` equals the four channels added, to float
                       precision, on every source.
    gas detector       `BldDataFEEGasDetEnergyV1`: six float64.

WHICH BLD SOURCE IS WHICH. A BLD payload's `src.phy` is the `BldInfo` enum
value, confirmed by the three that are unambiguous (EBeam = 0, PhaseCavity = 1,
FEEGasDetEnergy = 2 all match their payload type). The four `IpmFexV1` sources
are NOT identified by hardcoding the rest of that enum, which would be
guesswork. They are identified by FINGERPRINT: DATA.md records, for run 389,
each monitor's mean ratio between VCC-open and VCC-closed shots (lombpm 2.412,
diodeU 1.668, lomdiode 0.983, diode2 0.978) and the median of the specific
channels the lab uses. `identify_monitors` computes both from the stream and
matches. That makes the naming a measurement with a residual, not an assumption
-- and it doubles as an end-to-end check that the branch decode and the diode
decode are mutually consistent.

Run:  python -m automask.io.xtc_raw 389            # decode + validate + report
      python -m automask.io.xtc_raw 389 --events 2000
"""
from __future__ import annotations

import glob
import os
import struct
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
XTC_DIR = os.path.join(ROOT, "xtc")

DGRAM_HEADER = 40          # Sequence(16) + Env(4) + Xtc(20)
XTC_HEADER = 20            # Damage(4) + Src(8) + TypeId(4) + extent(4)
INVALID_FIDUCIAL = 0x1FFFF

TID_XTC = 1
TID_GASDET = 14
TID_EVR = 19
TID_IPMFEX = 31
TID_ANALOG = 94

#: `BldInfo` enum values whose payload type makes them self-identifying.
BLD_KNOWN = {0: "EBeam", 1: "PhaseCavity", 2: "FEEGasDetEnergy"}

#: Per-monitor fingerprints from `docs/DATA.md`, run 389: the ratio of the
#: monitor's mean between VCC-open and VCC-closed shots. The two that follow the
#: branch are the informative ones -- they are far enough apart (2.41 vs 1.67)
#: to separate, and far enough from the upstream monitors (~0.98) that a
#: mis-assignment cannot hide.
BRANCH_RATIO_389 = {"lombpm": 2.412, "diodeU": 1.668,
                    "lomdiode": 0.983, "diode2": 0.978}

#: Median of the channel the lab reads, run 389 (`docs/DATA.md`), as a second,
#: independent discriminant.
CHANNEL_MEDIAN_389 = {"diodeU": (0, 0.0178), "lombpm": (2, 0.2065),
                      "diode2": (0, 0.0539)}

BEAM_ON_CODE = 137
AIN_CC_CHANNEL = 2
AIN_VCC_CHANNEL = 3
CC_VCC_THRESHOLD = 2.0


# ==========================================================================
#  container
# ==========================================================================
def stream_files(run: int) -> List[str]:
    files = sorted(glob.glob(os.path.join(XTC_DIR, f"xppl1016922-r{run:04d}-s*-c00.xtc")))
    if not files:
        raise FileNotFoundError(f"no XTC streams for run {run} in {XTC_DIR}")
    return files


def _walk(fh, start: int, end: int, want: frozenset, depth: int = 0,
          maxdepth: int = 6) -> Iterator[Tuple[int, int, int, bytes]]:
    """Yield `(typeid, version, src_phy, payload)` for wanted leaves in [start, end).

    Walks by SEEKING rather than by loading the datagram: a single event carries
    ~4 MB of detector frames next to a few hundred bytes of scalars, so reading
    payloads indiscriminately would move the whole 2.9 GB file to answer a
    question about diodes.
    """
    off = start
    while off + XTC_HEADER <= end:
        fh.seek(off)
        head = fh.read(XTC_HEADER)
        if len(head) < XTC_HEADER:
            return
        _damage, _src_log, src_phy, contains, extent = struct.unpack("<5I", head)
        if extent < XTC_HEADER or off + extent > end:
            return                       # truncated stream: stop, do not guess
        tid = contains & 0xFFFF
        ver = (contains >> 16) & 0x7FFF
        if tid == TID_XTC and depth < maxdepth:
            yield from _walk(fh, off + XTC_HEADER, off + extent, want, depth + 1)
        elif tid in want:
            fh.seek(off + XTC_HEADER)
            yield tid, ver, src_phy, fh.read(extent - XTC_HEADER)
        off += extent


def iter_events(path: str, want: Sequence[int], max_events: Optional[int] = None):
    """Yield `(fiducial, clock_seconds, {(tid, src): payload})` per L1Accept.

    Transitions (Configure, BeginRun, Enable, ...) are skipped by their invalid
    fiducial. They matter: the Configure datagram carries a BLD payload full of
    uninitialised memory, which decodes to float64 values around 1e154 and would
    otherwise enter the monitor arrays as a first "shot".
    """
    want = frozenset(want)
    size = os.path.getsize(path)
    n = 0
    with open(path, "rb") as fh:
        off = 0
        while off + DGRAM_HEADER <= size:
            fh.seek(off)
            head = fh.read(DGRAM_HEADER)
            if len(head) < DGRAM_HEADER:
                break
            nsec, sec, _ticks, stamp, _env, _dmg, _sl, _sp, _cont, extent = \
                struct.unpack("<10I", head)
            if extent < XTC_HEADER:
                break
            end = off + 20 + extent
            if end > size:
                break                    # truncated final datagram
            fid = stamp & INVALID_FIDUCIAL
            if fid != INVALID_FIDUCIAL:
                payloads = {(tid, sp): p
                            for tid, _v, sp, p in _walk(fh, off + DGRAM_HEADER, end, want)}
                if payloads:
                    yield fid, sec + nsec * 1e-9, payloads
                    n += 1
                    if max_events and n >= max_events:
                        return
            off = end


def run_number(path: str) -> Optional[int]:
    """Run number from the BeginRun transition's `env` word, for cross-checking
    the filename."""
    with open(path, "rb") as fh:
        off = 0
        for _ in range(8):
            fh.seek(off)
            head = fh.read(DGRAM_HEADER)
            if len(head) < DGRAM_HEADER:
                return None
            *_, env, _d, _sl, _sp, _c, extent = struct.unpack("<10I", head)
            if 0 < env < 10000:
                return int(env)
            off += 20 + extent
    return None


# ==========================================================================
#  payload decoders
# ==========================================================================
def decode_evr(payload: bytes) -> List[int]:
    n = struct.unpack_from("<I", payload, 0)[0]
    if 4 + 12 * n > len(payload):
        return []
    return [struct.unpack_from("<3I", payload, 4 + 12 * i)[2] for i in range(n)]


def decode_analog(payload: bytes) -> np.ndarray:
    n = struct.unpack_from("<I", payload, 0)[0]
    if not (0 < n <= 64) or 4 + 8 * n > len(payload):
        return np.full(16, np.nan)
    return np.asarray(struct.unpack_from(f"<{n}d", payload, 4), dtype=np.float64)


def decode_ipmfex(payload: bytes) -> Tuple[np.ndarray, float, float, float]:
    ch0, ch1, ch2, ch3, total, xpos, ypos = struct.unpack_from("<7f", payload, 0)
    return np.array([ch0, ch1, ch2, ch3]), total, xpos, ypos


def decode_gasdet(payload: bytes) -> float:
    """`f_11_ENRC`, the channel `read_xtc.scan_shots` uses."""
    return float(struct.unpack_from("<d", payload, 0)[0])


# ==========================================================================
#  a scan of one run
# ==========================================================================
@dataclass
class RawScan:
    """Everything decoded from one stream, before any monitor is given a name."""
    run: int
    files: List[str]
    fiducial: np.ndarray
    clock: np.ndarray
    beam_on: np.ndarray
    cc_open: np.ndarray
    vcc_open: np.ndarray
    volts: np.ndarray                        # (n, 16)
    ipm_sum: Dict[int, np.ndarray] = field(default_factory=dict)
    ipm_channels: Dict[int, np.ndarray] = field(default_factory=dict)
    gasdet: Optional[np.ndarray] = None
    codes_seen: Tuple[int, ...] = ()

    @property
    def n_events(self) -> int:
        return int(self.fiducial.size)


def scan(run: int, max_events: Optional[int] = None, verbose: bool = True) -> RawScan:
    """Decode the per-shot scalars of every local stream of `run`."""
    files = stream_files(run)
    want = (TID_EVR, TID_ANALOG, TID_IPMFEX, TID_GASDET)

    fids, clocks, beam, volts, gas = [], [], [], [], []
    sums: Dict[int, list] = {}
    chans: Dict[int, list] = {}
    codes_seen = set()

    for path in files:
        rn = run_number(path)
        if rn is not None and rn != run:
            raise RuntimeError(f"{os.path.basename(path)} declares run {rn}, not {run}")
        for fid, clock, payloads in iter_events(path, want, max_events):
            evr = next((p for (t, _s), p in payloads.items() if t == TID_EVR), None)
            ain = next((p for (t, _s), p in payloads.items() if t == TID_ANALOG), None)
            if evr is None or ain is None:
                continue
            cs = decode_evr(evr)
            codes_seen.update(cs)
            v = decode_analog(ain)
            fids.append(fid)
            clocks.append(clock)
            beam.append(BEAM_ON_CODE in cs)
            volts.append(v)
            for (tid, sp), p in payloads.items():
                if tid == TID_IPMFEX:
                    ch, total, _x, _y = decode_ipmfex(p)
                    sums.setdefault(sp, []).append(total)
                    chans.setdefault(sp, []).append(ch)
                elif tid == TID_GASDET:
                    gas.append(decode_gasdet(p))
        if verbose:
            print(f"[xtc_raw] {os.path.basename(path)}: {len(fids)} events so far")

    if not fids:
        raise RuntimeError(f"run {run}: no L1Accept datagrams decoded")
    volts = np.asarray(volts)
    n = len(fids)
    # A source missing from some events would silently misalign every array, so
    # only sources present on EVERY decoded event are kept.
    keep = {sp for sp, v in sums.items() if len(v) == n}
    if verbose and set(sums) - keep:
        print(f"[xtc_raw] dropped ragged BLD sources: "
              f"{sorted(hex(s) for s in set(sums) - keep)}")
    return RawScan(
        run=run, files=files,
        fiducial=np.asarray(fids, dtype=np.int64),
        clock=np.asarray(clocks, dtype=np.float64),
        beam_on=np.asarray(beam, dtype=bool),
        cc_open=volts[:, AIN_CC_CHANNEL] > CC_VCC_THRESHOLD,
        vcc_open=volts[:, AIN_VCC_CHANNEL] > CC_VCC_THRESHOLD,
        volts=volts,
        ipm_sum={sp: np.asarray(sums[sp]) for sp in keep},
        ipm_channels={sp: np.stack(chans[sp]) for sp in keep},
        gasdet=np.asarray(gas) if len(gas) == n else None,
        codes_seen=tuple(sorted(codes_seen)))


# ==========================================================================
#  naming the monitors, by fingerprint
# ==========================================================================
#: Only the two monitors DOWNSTREAM of the CC/VCC split can be identified this
#: way, and they are the only two the masking project uses. `lomdiode` and
#: `diode2` sit upstream, so their branch ratios are 0.98 and 0.98 -- identical
#: to within the noise of any run, and the matcher that tried to separate them
#: assigned them oppositely on runs 389 and 396. Naming a source the evidence
#: cannot distinguish is worse than leaving it numbered.
BRANCH_FOLLOWING = ("lombpm", "diodeU")
MIN_BRANCH_RATIO = 1.2


def identify_monitors(scan: RawScan, verbose: bool = True) -> Dict[str, int]:
    """Map `lombpm` and `diodeU` to BLD source ids by their branch response.

    The split is physically visible: DATA.md records that these two monitors
    change by 2.41x and 1.67x between VCC-open and VCC-closed shots while every
    upstream monitor changes by ~2%. That is a large, ordered, self-contained
    signature, and matching it identifies the two sources without hardcoding the
    `BldInfo` enum. The lab's absolute channel medians are then a SECOND,
    independent confirmation, printed next to the value they should match.

    Sources whose branch ratio is below `MIN_BRANCH_RATIO` are upstream and are
    left unnamed -- see `BRANCH_FOLLOWING`.
    """
    lit = scan.beam_on
    on, off = lit & scan.vcc_open, lit & ~scan.vcc_open
    rows = []
    for sp, s in sorted(scan.ipm_sum.items()):
        if on.sum() < 20 or off.sum() < 20:
            ratio = np.nan
        else:
            denom = float(np.mean(s[off]))
            ratio = float(np.mean(s[on]) / denom) if denom else np.nan
        rows.append((sp, ratio, float(np.median(s[lit]))))

    assigned: Dict[str, int] = {}
    used = set()
    for name in sorted(BRANCH_FOLLOWING, key=lambda n: -BRANCH_RATIO_389[n]):
        target = BRANCH_RATIO_389[name]
        best, err = None, np.inf
        for sp, ratio, _m in rows:
            if sp in used or not np.isfinite(ratio) or ratio < MIN_BRANCH_RATIO:
                continue
            e = abs(ratio - target) / target
            if e < err:
                best, err = sp, e
        if best is not None and err < 0.3:
            assigned[name] = best
            used.add(best)

    if verbose:
        print("\n  BLD source   branch ratio   median(sum)   identified as")
        for sp, ratio, med in rows:
            name = next((k for k, v in assigned.items() if v == sp), None)
            tag = (f"{name}  (DATA.md: {BRANCH_RATIO_389[name]:.3f})" if name
                   else "upstream -- not branch-separable, left unnamed")
            print(f"    0x{sp:02x}       {ratio:8.3f}   {med:11.5f}   {tag}")
        for name, (ci, med) in CHANNEL_MEDIAN_389.items():
            if name in assigned:
                got = float(np.median(scan.ipm_channels[assigned[name]][lit, ci]))
                print(f"    confirm: {name}/channels[{ci}] median {got:.4f} "
                      f"(DATA.md run 389: {med:.4f})")
    return assigned


def to_shot_meta(scan: RawScan, assigned: Optional[Dict[str, int]] = None):
    """A `ShotMeta` from a raw scan -- the same object `read_xtc.scan_shots`
    returns, so `ShotSelection` and everything downstream is unchanged."""
    from automask.shot_selection import ShotMeta

    assigned = identify_monitors(scan, verbose=False) if assigned is None else assigned
    intensity: Dict[str, np.ndarray] = {}
    if "diodeU" in assigned:
        sp = assigned["diodeU"]
        intensity["diodeU"] = scan.ipm_sum[sp]
        intensity["sample_diode"] = scan.ipm_channels[sp][:, 0]
    if "lombpm" in assigned:
        intensity["lombpm"] = scan.ipm_sum[assigned["lombpm"]]
    if scan.gasdet is not None:
        intensity["gasdet"] = scan.gasdet
    if not intensity:
        raise RuntimeError("no monitor could be identified; refusing to build a "
                           "ShotMeta with unnamed sources")
    return ShotMeta(run=scan.run, beam_on=scan.beam_on, cc_open=scan.cc_open,
                    vcc_open=scan.vcc_open, intensity=intensity)


def scan_shots(run: int, max_events: Optional[int] = None):
    """psana-free drop-in for `read_xtc.scan_shots`."""
    return to_shot_meta(scan(run, max_events))


# ==========================================================================
#  validation
# ==========================================================================
def validate(scan: RawScan) -> Dict[str, float]:
    """Check every decode against something known independently of it."""
    out: Dict[str, float] = {}
    out["n_events"] = float(scan.n_events)
    out["frac_beam_on"] = float(scan.beam_on.mean())

    v2, v3 = scan.volts[:, AIN_CC_CHANNEL], scan.volts[:, AIN_VCC_CHANNEL]
    rail = ((np.abs(v2) < 0.5) | (np.abs(v2 - 5.05) < 0.5))
    rail &= ((np.abs(v3) < 0.5) | (np.abs(v3 - 5.05) < 0.5))
    out["frac_shutters_on_rail"] = float(rail.mean())
    out["n_channels"] = float(scan.volts.shape[1])

    err = []
    for sp, s in scan.ipm_sum.items():
        err.append(float(np.max(np.abs(s - scan.ipm_channels[sp].sum(axis=1)))))
    out["max_ipm_sum_residual"] = max(err) if err else float("nan")

    out["fiducial_monotonic_frac"] = float(np.mean(np.diff(scan.fiducial) > 0))
    out["clock_monotonic_frac"] = float(np.mean(np.diff(scan.clock) >= 0))
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", type=int, nargs="*", default=None)
    ap.add_argument("--events", type=int, default=None)
    args = ap.parse_args()

    runs = args.runs or [int(os.path.basename(p).split("-r")[1][:4])
                         for p in sorted(glob.glob(
                             os.path.join(XTC_DIR, "xppl1016922-r*-s*-c00.xtc")))]
    for run in sorted(set(runs)):
        print(f"\n{'=' * 70}\nrun {run}")
        s = scan(run, args.events)
        v = validate(s)
        print(f"  events {int(v['n_events'])}, beam on {100*v['frac_beam_on']:.1f}%, "
              f"EVR codes {list(s.codes_seen)}")
        print(f"  analog input: {int(v['n_channels'])} channels, "
              f"{100*v['frac_shutters_on_rail']:.1f}% of shots have ch02/ch03 on a rail")
        print(f"  IpmFex sum vs its own channels: max residual "
              f"{v['max_ipm_sum_residual']:.2e}")
        print(f"  fiducial increasing on {100*v['fiducial_monotonic_frac']:.1f}% of "
              f"steps, clock on {100*v['clock_monotonic_frac']:.1f}%")
        print(f"  CC open {100*s.cc_open.mean():.1f}%, VCC open {100*s.vcc_open.mean():.1f}%")
        identify_monitors(s)


if __name__ == "__main__":
    main()
