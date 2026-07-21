"""
lcls_xpp.py -- Read the local xppl1016922 (LCLS / XPP) data without psana.

Everything you need from this experiment is reachable from two kinds of files:

  * hdf5/smalldata/xppl1016922_Run<NNNN>.h5   -- per-event reduced "small data"
  * calib/.../<START>-end.data                -- psana detector calibration constants

This module gives you plain numpy / dict access to both, plus helpers to
assemble a raw panel-stack into a 2D image. No psana, no LCLS cluster needed.

Only dependency: numpy + h5py  (pip install h5py).

--------------------------------------------------------------------------
The ":" -> U+F022 filename quirk
--------------------------------------------------------------------------
psana calib directory names contain ":" (e.g. "Epix100a::CalibV1",
"XppGon.0:Epix100a.1").  This copy was made on macOS, which cannot store ":"
in a filename, so every ":" was replaced by the private-use character U+F022.
`resolve()` below rewrites ":" -> U+F022 for you, so you can keep writing the
natural psana names in your code.
"""

from __future__ import annotations
import os
import glob
import numpy as np
import h5py

# ---- paths -----------------------------------------------------------------
# Root of the local data copy (the dir that contains xtc/, hdf5/, calib/ ...).
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/io/ -> LCLS
COLON = chr(0xF022)          # the char that stands in for ":" on disk


def resolve(path: str) -> str:
    """Turn a natural psana path (with ':') into the on-disk path (U+F022)."""
    return path.replace(":", COLON)


def smalldata_path(run: int) -> str:
    return os.path.join(ROOT, "hdf5", "smalldata", f"xppl1016922_Run{run:04d}.h5")


# ===========================================================================
#  Small-data HDF5
# ===========================================================================
class SmallData:
    """Thin, friendly wrapper over one xppl1016922_Run<NNNN>.h5 file.

    The file has two logically different parts:

      * per-event arrays   (first axis == number of events), e.g.
            ipm2/sum, ebeam/photon_energy, lightStatus/laser,
            jungfrau1M_alcove/azav_azav (the pre-computed azimuthal average)
      * configuration       under  UserDataCfg/...   (geometry, masks, calib
            constants, azimuthal-integration parameters -- written once)
      * run sums            under  Sums/...   (detector image summed over the run)

    Usage
    -----
    >>> sd = SmallData(475)
    >>> sd.nevents
    3201
    >>> e = sd.photon_energy_keV()          # per event
    >>> geo = sd.jungfrau_geometry()        # dict of geometry / masks / calib
    >>> img = sd.sum_image()                # assembled 2D run-sum image
    """

    def __init__(self, run: int, path: str | None = None):
        self.run = run
        self.path = path or smalldata_path(run)
        if not os.path.exists(self.path):
            raise FileNotFoundError(self.path)
        self.h5 = h5py.File(self.path, "r")

    # -- context manager sugar ------------------------------------------
    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def close(self):
        self.h5.close()

    # -- generic access -------------------------------------------------
    @property
    def nevents(self) -> int:
        return int(self.h5["fiducials"].shape[0])

    def get(self, key: str) -> np.ndarray:
        """Read any dataset by its full path, e.g. 'ipm2/sum'."""
        return self.h5[key][()]

    def keys(self, group: str = "/"):
        return list(self.h5[group].keys())

    def tree(self, group: str = "/", _pre: str = "") -> None:
        """Print the group/dataset tree (with shapes)."""
        for k in sorted(self.h5[group].keys()):
            item = self.h5[f"{group.rstrip('/')}/{k}"]
            if isinstance(item, h5py.Group):
                print(f"{_pre}{k}/")
                self.tree(f"{group.rstrip('/')}/{k}", _pre + "    ")
            else:
                print(f"{_pre}{k}  {item.shape} {item.dtype}")

    # -- common per-event quantities ------------------------------------
    def event_time(self) -> np.ndarray:
        return self.h5["event_time"][()]

    def laser_on(self) -> np.ndarray:
        """1 where the optical laser (pump) fired, else 0."""
        return self.h5["lightStatus/laser"][()].astype(bool)

    def xray_on(self) -> np.ndarray:
        return self.h5["lightStatus/xray"][()].astype(bool)

    def photon_energy_keV(self) -> np.ndarray:
        """Per-event FEL photon energy from the electron beam, in keV."""
        return self.h5["ebeam/photon_energy"][()]

    def i0(self, monitor: str = "ipm2") -> np.ndarray:
        """Incident-intensity monitor sum (default XPP-SB2 BMMON = ipm2)."""
        return self.h5[f"{monitor}/sum"][()]

    def scan_value(self):
        """The scanned variable per event, or None if this run is static.

        Returns (name, values) or None.  For run 475 the run is static
        (varStep is all zeros) -- the scan is encoded elsewhere / fixed.
        """
        var = self.h5["scan/varStep"][()]
        if np.unique(var).size <= 1:
            return None
        return ("scan/varStep", var)

    def epics_once(self, name: str):
        """A 'written once per run' EPICS PV (motor position, energy, ...).

        e.g. epics_once('lom_E')  -> mono photon energy in keV
             epics_once('delay')  -> pump-probe delay
        """
        return self.h5[f"epicsOnce/{name}"][()]

    # -- Jungfrau geometry + calibration --------------------------------
    def jungfrau_geometry(self, det: str = "jungfrau1M_alcove") -> dict:
        """Everything needed to work with / integrate the Jungfrau1M.

        Returns a dict with (arrays are in the native panel stack shape
        (2, 512, 1024) unless noted):

          pixel_size_m   scalar, 7.5e-5
          img_shape      (H, W) of the assembled 2D image
          ix, iy         per-pixel column/row index into the assembled image
          x, y, z        per-pixel lab-frame coordinates [micron]
          mask           1 = good pixel, 0 = bad (user + status combined)
          cmask          calibration/status-only mask
          ped, rms, gain, offset   per-gain-stage calib (3, 2, 512, 1024)
          pixel_status   per-gain-stage status
          # azimuthal-integration parameters used by smalldata_tools:
          dist_mm        sample->detector distance [mm]      (190.0)
          wavelength_A   X-ray wavelength [Angstrom]         (1.2915)
          energy_keV     photon energy [keV]                 (9.6)
          beam_center_mm (xcen, ycen) of the direct beam in the det plane [mm]
          q              (nq,) q of each azimuthal bin [1/Angstrom]
          q_bin_edges    (nq+1,) bin edges
          adu_per_photon scalar
        """
        g = self.h5[f"UserDataCfg/{det}"]

        def a(k):
            return g[k][()]

        def s(k):
            return float(np.ravel(g[k][()])[0])

        img_shape = tuple(int(v) for v in a("imgShape"))
        # imgShape is stored as (nx, ny); the assembled array is (ny, nx).
        iy = a("iy")
        ix = a("ix")
        H, W = int(iy.max()) + 1, int(ix.max()) + 1

        xcen = s("azav__azav_xcen")   # mm
        ycen = s("azav__azav_ycen")   # mm

        return dict(
            pixel_size_m=s("pixelsize"),
            img_shape=(H, W),
            img_shape_cfg=img_shape,
            ix=ix, iy=iy,
            x=a("x"), y=a("y"), z=a("z"),
            mask=a("mask"), cmask=a("cmask"),
            status_mask=a("statusMask"),
            ped=a("ped"), rms=a("rms"), gain=a("gain"),
            offset=a("offset"), pixel_status=a("pixel_status"),
            dist_mm=s("azav__azav_dis_to_sam"),
            wavelength_A=s("azav__azav_lam"),
            energy_keV=s("azav__azav_eBeam"),
            beam_center_mm=(xcen, ycen),
            q=a("azav__azav_q"),
            q_bin_edges=a("azav__azav_qbins"),
            q_bin_width=s("azav__azav_qbin"),
            adu_per_photon=s("azav__azav_ADU_per_photon"),
            header=g["azav__azav_header"][()][0].decode(),
        )

    def assemble(self, panelstack: np.ndarray, det: str = "jungfrau1M_alcove",
                 fill: float = 0.0) -> np.ndarray:
        """Turn a native panel stack (2, 512, 1024) into a 2D image (H, W).

        Uses the ix/iy pixel maps stored in the file -- the same mapping
        smalldata_tools / psana uses, so the result matches the real
        detector layout.
        """
        g = self.h5[f"UserDataCfg/{det}"]
        ix = g["ix"][()]
        iy = g["iy"][()]
        H, W = int(iy.max()) + 1, int(ix.max()) + 1
        out = np.full((H, W), fill, dtype=np.asarray(panelstack).dtype)
        out[iy.ravel(), ix.ravel()] = np.asarray(panelstack).ravel()
        return out

    def sum_image(self, det: str = "jungfrau1M_alcove",
                  which: str = "calib") -> np.ndarray:
        """Assembled 2D image of the run-summed detector.

        `which` selects the Sums dataset: 'calib' = sum of calibrated frames.
        This is a real assembled 2D detector image, ready for your own analysis.
        """
        stack = self.h5[f"Sums/{det}_{which}"][()]
        return self.assemble(stack, det=det)

    def azav(self, det: str = "jungfrau1M_alcove"):
        """The azimuthal average smalldata_tools already computed.

        Returns (q, I) where q is (nq,) [1/Angstrom] and I is
        (nevents, nphi, nq).  For a quick radial profile average over events
        with signal.  Handy as a reference radial profile / cross-check.
        """
        q = self.h5[f"UserDataCfg/{det}/azav__azav_q"][()]
        I = self.h5[f"{det}/azav_azav"][()]
        return q, I


# ===========================================================================
#  psana calib "<START>-end.data" files
# ===========================================================================
def calib_data_dir(dettype: str, source: str, ctype: str) -> str:
    """Path to a calibration-type dir, e.g.
       calib_data_dir('Epix100a::CalibV1', 'XppGon.0:Epix100a.1', 'pedestals')
    """
    return resolve(os.path.join(ROOT, "calib", dettype, source, ctype))


def list_calib(dettype: str, source: str, ctype: str):
    """List available run-range files for a constant type, newest first.

    Returns list of (start_run, path).  A file '<START>-end.data' applies
    from START onward until superseded by a higher START.
    """
    d = calib_data_dir(dettype, source, ctype)
    out = []
    for p in glob.glob(os.path.join(d, "*-end.data")):
        start = int(os.path.basename(p).split("-")[0])
        out.append((start, p))
    return sorted(out, reverse=True)


def load_calib(dettype: str, source: str, ctype: str, run: int) -> np.ndarray:
    """Load the calibration constant valid for `run`.

    The psana .data files are ASCII arrays (numpy-loadable) with a '#'-comment
    header.  This picks the correct run-range file, loads it, and reshapes it
    to the detector's natural shape based on the header NDARRAY dims when
    present (otherwise returns the 2D array numpy sees).
    """
    files = list_calib(dettype, source, ctype)
    chosen = None
    for start, p in files:               # files are newest-first
        if start <= run:
            chosen = p
            break
    if chosen is None:                    # fall back to lowest available
        chosen = files[-1][1]
    arr = np.loadtxt(chosen)
    dims = _ndarray_dims(chosen)
    if dims is not None and int(np.prod(dims)) == arr.size:
        arr = arr.reshape(dims)
    return arr


def _ndarray_dims(path: str):
    """Parse 'NDARRAY_DIMS' / 'DIMENSION' style header lines if any."""
    dims = None
    with open(path) as f:
        for line in f:
            if not line.startswith("#"):
                break