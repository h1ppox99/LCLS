"""Reusable helpers for the production notebook."""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np

from automask.io.lcls_xpp import COLON, ROOT, SmallData, smalldata_path
from automask.io.read_xtc import JUNGFRAU_NAME, XTC_DIR, calib_dir, open_local_run


_PAYLOAD_FIELDS = {
    "Jungfrau.ElementV2": ("frameNumber", "fiducials", "ticks"),
    "Bld.BldDataEBeamV7": (
        "ebeamCharge", "ebeamDumpCharge", "ebeamEnergyBC1",
        "ebeamEnergyBC2", "ebeamL3Energy", "ebeamLTU250", "ebeamLTU450",
        "ebeamLTUAngX", "ebeamLTUAngY", "ebeamLTUPosX", "ebeamLTUPosY",
        "ebeamPhotonEnergy", "ebeamPkCurrBC1", "ebeamPkCurrBC2",
        "ebeamUndAngX", "ebeamUndAngY", "ebeamUndPosX", "ebeamUndPosY",
        "ebeamXTCAVAmpl", "ebeamXTCAVPhase", "damageMask",
    ),
    "Bld.BldDataPhaseCavityV1": (
        "charge1", "charge2", "fitTime1", "fitTime2",
    ),
    "Bld.BldDataFEEGasDetEnergyV1": (
        "f_11_ENRC", "f_12_ENRC", "f_21_ENRC",
        "f_22_ENRC", "f_63_ENRC", "f_64_ENRC",
    ),
    "Lusi.IpmFexV1": ("channel", "sum", "xpos", "ypos"),
    "Bld.BldDataAnalogInputV1": ("channelVoltages",),
    "Bld.BldDataBeamMonitorV1": (
        "TotalIntensity", "X_Position", "Y_Position",
    ),
}


def _payload_type_name(payload_type):
    if payload_type is None:
        return "—"
    module = payload_type.__module__
    module = module[len("psana."):] if module.startswith("psana.") else module
    return f"{module}.{payload_type.__name__}"


def configure_psana_environment():
    """Set psana's required paths before psana is first imported."""
    repo_root = Path(__file__).resolve().parent.parent
    local_config = repo_root / "psana_env.local"
    config = {}
    if local_config.exists():
        for raw_line in local_config.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            config[key.strip()] = value.strip().strip(chr(34) + chr(39))

    psdm = os.environ.get("SIT_PSDM_DATA") or config.get("PSANA_PSDM")
    if not psdm:
        raise RuntimeError(
            f"Cannot configure psana: SIT_PSDM_DATA is unset and {local_config} "
            "does not define PSANA_PSDM. Configure psana_env.local as described "
            "in docs/PSANA_XTC.md, then restart the kernel."
        )
    psdm = str(Path(psdm).expanduser().resolve())
    os.environ["SIT_PSDM_DATA"] = psdm
    os.environ.setdefault("SIT_ROOT", str(Path(psdm) / "sit_root"))
    os.environ.setdefault("SIT_DATA", str(Path(psdm) / "data"))
    return {
        name: os.environ[name]
        for name in ("SIT_PSDM_DATA", "SIT_ROOT", "SIT_DATA")
    }


def _show_table(title, rows, columns=None):
    from IPython.display import Markdown, display

    display(Markdown(f'### {title}'))
    if not rows:
        display(Markdown('_None found._'))
        return
    columns = columns or list(rows[0])

    def clean(value):
        return str(value).replace('|', '\\|').replace('\n', '<br>')

    lines = [
        '| ' + ' | '.join(columns) + ' |',
        '| ' + ' | '.join('---' for _ in columns) + ' |',
    ]
    lines.extend(
        '| ' + ' | '.join(clean(row.get(column, '')) for column in columns) + ' |'
        for row in rows
    )
    display(Markdown('\n'.join(lines)))


def _human_bytes(n_bytes):
    value = float(n_bytes)
    for unit in ('B', 'KiB', 'MiB', 'GiB', 'TiB'):
        if value < 1024 or unit == 'TiB':
            return f'{value:.2f} {unit}'
        value /= 1024


def _xtc_inventory(experiment, run):
    pattern = re.compile(r'-r(?P<run>\d+)-s(?P<stream>\d+)-c(?P<chunk>\d+)\.xtc$')
    files = sorted(Path(XTC_DIR).glob(f'{experiment}-r{run:04d}-s*-c*.xtc'))
    rows = []
    for path in files:
        match = pattern.search(path.name)
        stream = int(match.group('stream')) if match else None
        chunk = int(match.group('chunk')) if match else None
        rows.append({
            'file': path.name,
            'stream': f's{stream:02d}' if stream is not None else '?',
            'chunk': f'c{chunk:02d}' if chunk is not None else '?',
            'size': _human_bytes(path.stat().st_size),
            'path': str(path.resolve()),
        })
    return {
        'files': rows,
        'total_bytes': sum(path.stat().st_size for path in files),
    }


def _calibration_file_inventory(run, detector_type, detector_source):
    source_dir = (
        Path(ROOT) / 'calib'
        / detector_type.replace(':', COLON)
        / detector_source.replace(':', COLON)
    )
    rows = []
    if not source_dir.exists():
        return rows
    for constant_dir in sorted(path for path in source_dir.iterdir() if path.is_dir()):
        candidates = []
        for path in constant_dir.glob('*.data'):
            match = re.fullmatch(r'(\d+)-(end|\d+)\.data', path.name)
            if not match:
                continue
            start = int(match.group(1))
            end = float('inf') if match.group(2) == 'end' else int(match.group(2))
            if start <= run <= end:
                candidates.append((start, end, path))
        selected = max(candidates, default=None, key=lambda item: item[0])
        if selected is None:
            continue
        start, end, path = selected
        rows.append({
            'constant': constant_dir.name,
            'run range': f"{start}–{'end' if end == float('inf') else int(end)}",
            'size': _human_bytes(path.stat().st_size),
            'path': str(path.resolve()),
        })
    return rows


def list_experiment_content(
    experiment, run, detector, detector_source, detector_calib_type
):
    from IPython.display import Markdown, display

    xtc = _xtc_inventory(experiment, run)
    calibration = _calibration_file_inventory(
        run, detector_calib_type, detector_source
    )
    display(Markdown(
        f"**Experiment:** `{experiment}`  \n"
        f"**Run:** `{run:04d}`  \n"
        f"**Detector:** `{detector}` (`{detector_source}`)"
    ))
    _show_table(
        "Relevant XTC files",
        xtc["files"],
        ["file", "stream", "chunk", "size", "path"],
    )
    _show_table(
        "Applicable detector calibration files",
        calibration,
        ["constant", "run range", "size", "path"],
    )
    return {
        "experiment": experiment,
        "run": int(run),
        "detector": detector,
        "detector_source": detector_source,
        "detector_calib_type": detector_calib_type,
        "xtc": xtc,
        "calibration": calibration,
    }


def _payload_value(payload, accessor):
    value = getattr(payload, accessor)()
    array = np.asarray(value)
    if array.ndim == 0:
        return {accessor: float(array)}
    if array.ndim != 1 or array.size > 64 or array.dtype.kind not in "biuf":
        raise ValueError("field is not a short numeric vector")
    return {
        f"{accessor}[{index}]": float(item)
        for index, item in enumerate(array)
    }


def _column_summary(values):
    if values.dtype.kind not in "biufc":
        available_mask = np.asarray([value is not None for value in values])
        available = int(available_mask.sum())
        if not available:
            return "0.0%", "unavailable", "no values"
        observed = values[available_mask].astype(str)
        unique, counts = np.unique(observed, return_counts=True)
        coverage = f"{available / values.size:.1%}"
        if unique.size == 1:
            return coverage, "constant", unique[0]
        changes = int(np.count_nonzero(
            available_mask[1:] & available_mask[:-1]
            & (values[1:] != values[:-1])
        ))
        counts_text = ", ".join(
            f"{value}: {count}" for value, count in zip(unique, counts)
        )
        return coverage, f"{unique.size} values; {changes} changes", counts_text

    finite = np.isfinite(values)
    available = int(finite.sum())
    if not available:
        return "0.0%", "unavailable", "no finite values"
    observed = values[finite]
    unique = np.unique(observed)
    coverage = f"{available / values.size:.1%}"
    if unique.size == 1:
        return coverage, "constant", f"{unique[0]:.4g}"
    if unique.size <= 10:
        changes = int(np.count_nonzero(
            finite[1:] & finite[:-1] & (values[1:] != values[:-1])
        ))
        counts = ", ".join(
            f"{value:.8g}: {int(np.count_nonzero(observed == value))}"
            for value in unique
        )
        return coverage, f"{unique.size} values; {changes} changes", counts
    p05, median, p95 = np.percentile(observed, [5, 50, 95])
    return (
        coverage,
        f"continuous; {unique.size} values",
        f"p05={p05:.4g}, median={median:.4g}, p95={p95:.4g}",
    )


def profile_run_values(run):
    """Profile shot payloads and EPICS state over a full XTC run."""
    data_source, _ = open_local_run(run)
    epics_store = data_source.env().epicsStore()
    payloads = {}
    columns = {}
    field_metadata = {}
    evr_history = {}
    epics_columns = {}
    epics_metadata = {}
    n_events = 0

    for event in data_source.events():
        n_events += 1
        for values in columns.values():
            values.append(np.nan)
        for history in evr_history.values():
            history.append(None)
        for values in epics_columns.values():
            values.append(None)

        for event_key in event.keys():
            payload_type = event_key.type()
            type_name = _payload_type_name(payload_type)
            source_name = str(event_key.src())
            alias = event_key.alias() or "—"
            key_name = event_key.key() or ""
            identity = (source_name, type_name, key_name)
            info = payloads.setdefault(identity, {
                "source": source_name,
                "alias": alias,
                "type": type_name,
                "key": key_name or "—",
                "events": 0,
                "fields": set(),
                "errors": 0,
            })
            info["events"] += 1
            if (
                payload_type is None
                or (
                    type_name != "EvrData.DataV4"
                    and type_name not in _PAYLOAD_FIELDS
                )
            ):
                continue

            args = (payload_type, event_key.src())
            if key_name:
                args += (key_name,)
            payload = event.get(*args)
            if payload is None:
                continue

            if type_name == "EvrData.DataV4":
                if identity not in evr_history:
                    evr_history[identity] = [None] * n_events
                try:
                    evr_history[identity][-1] = {
                        fifo.eventCode() for fifo in payload.fifoEvents()
                    }
                except (AttributeError, TypeError, ValueError):
                    info["errors"] += 1
                continue

            for accessor in _PAYLOAD_FIELDS.get(type_name, ()):
                try:
                    extracted = _payload_value(payload, accessor)
                except (AttributeError, IndexError, TypeError, ValueError):
                    info["errors"] += 1
                    continue
                for field, value in extracted.items():
                    field_id = f"{source_name}/{type_name}/{field}"
                    if field_id not in columns:
                        columns[field_id] = [np.nan] * n_events
                        field_metadata[field_id] = {
                            "source": source_name,
                            "type": type_name,
                            "field": field,
                        }
                    columns[field_id][-1] = value
                    info["fields"].add(field)

        for pv_name in epics_store.pvNames():
            alias = epics_store.alias(pv_name) or pv_name
            field_id = f"EPICS/{alias}"
            if field_id not in epics_columns:
                epics_columns[field_id] = [None] * n_events
                epics_metadata[field_id] = {
                    "alias": alias,
                    "pv": pv_name,
                    "errors": 0,
                }
            try:
                value = epics_store.value(alias)
                array = np.asarray(value)
                if array.ndim != 0 or array.dtype.kind not in "biufUOS":
                    raise ValueError("EPICS value is not scalar")
                epics_columns[field_id][-1] = array.item()
            except (AttributeError, KeyError, TypeError, ValueError):
                epics_metadata[field_id]["errors"] += 1

    if not n_events:
        raise RuntimeError(f"Run {run:04d} contains no decodable events")

    for identity, history in evr_history.items():
        codes = sorted(set().union(*(codes or set() for codes in history)))
        source_name, type_name, _key_name = identity
        info = payloads[identity]
        for code in codes:
            field = f"eventCode[{code}]"
            field_id = f"{source_name}/{type_name}/{field}"
            columns[field_id] = [
                np.nan if event_codes is None else float(code in event_codes)
                for event_codes in history
            ]
            field_metadata[field_id] = {
                "source": source_name,
                "type": type_name,
                "field": field,
            }
            info["fields"].add(field)

    value_arrays = {
        field_id: np.asarray(values, dtype=float)
        for field_id, values in columns.items()
    }
    epics_arrays = {}
    for field_id, values in epics_columns.items():
        present = [value for value in values if value is not None]
        numeric = present and all(
            isinstance(value, (bool, int, float, np.number)) for value in present
        )
        epics_arrays[field_id] = np.asarray(
            [np.nan if value is None else value for value in values],
            dtype=float if numeric else object,
        )
    value_arrays.update(epics_arrays)
    payload_rows = []
    for info in sorted(
        payloads.values(), key=lambda row: (row["source"], row["type"], row["key"])
    ):
        payload_rows.append({
            "source": info["source"],
            "alias": info["alias"],
            "type": info["type"],
            "key": info["key"],
            "coverage": f'{info["events"] / n_events:.1%}',
            "profiled fields": len(info["fields"]),
            "read errors": info["errors"],
        })

    value_rows = []
    for field_id, values in sorted(
        ((field_id, values) for field_id, values in value_arrays.items()
         if not field_id.startswith("EPICS/"))
    ):
        coverage, variation, observed = _column_summary(values)
        metadata = field_metadata[field_id]
        value_rows.append({
            "source": metadata["source"],
            "field": metadata["field"],
            "type": metadata["type"],
            "coverage": coverage,
            "variation": variation,
            "observed": observed,
        })

    epics_rows = []
    for field_id, values in sorted(epics_arrays.items()):
        coverage, variation, observed = _column_summary(values)
        metadata = epics_metadata[field_id]
        epics_rows.append({
            "alias": metadata["alias"],
            "PV": metadata["pv"],
            "dtype": str(values.dtype),
            "coverage": coverage,
            "variation": variation,
            "observed": observed,
            "read errors": metadata["errors"],
        })

    _show_table(f"Raw XTC payloads ({n_events:,} decoded events)", payload_rows)
    _show_table("Automatically profiled per-shot values", value_rows)
    _show_table("EPICS process variables", epics_rows)
    return {
        "run": int(run),
        "events": n_events,
        "payloads": payload_rows,
        "values": value_arrays,
        "summary": {"xtc": value_rows, "epics": epics_rows},
        "epics": epics_rows,
    }


def print_detector_geometry(run, detector_name=JUNGFRAU_NAME):
    import psana

    data_source, _ = open_local_run(run)
    event = next(data_source.events(), None)
    if event is None:
        raise RuntimeError(f"Run {run:04d} contains no decodable events")
    detector = psana.Detector(detector_name)
    raw = detector.raw(event)
    calibrated = detector.calib(event)
    ix = detector.indexes_x(run)
    iy = detector.indexes_y(run)
    geometry = {
        "detector": detector_name,
        "psana calibration path": calib_dir(),
        "psana calibration path exists": os.path.isdir(calib_dir()),
        "native shape": tuple(raw.shape) if raw is not None else None,
        "raw dtype": str(raw.dtype) if raw is not None else None,
        "calibrated frame available": calibrated is not None,
        "index-map shape": tuple(ix.shape) if ix is not None else None,
        "assembled shape": (
            (int(iy.max()) + 1, int(ix.max()) + 1)
            if ix is not None and iy is not None else None
        ),
    }
    rows = [{"property": key, "value": value} for key, value in geometry.items()]
    _show_table("Jungfrau geometry", rows)


def print_small_data_inventory(run):
    path = Path(smalldata_path(run))
    summary = {
        "file": path.name,
        "exists": path.exists(),
        "size": _human_bytes(path.stat().st_size) if path.exists() else "—",
        "events": "—",
        "top-level groups": "—",
    }
    if not path.exists():
        return summary
    with SmallData(run) as small_data:
        summary["events"] = small_data.nevents
        summary["top-level groups"] = ", ".join(small_data.keys())
    _show_table("Small-data content", [summary])
    return summary
