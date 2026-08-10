"""Reusable helpers for the production notebook."""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np

from automask.io.lcls_xpp import COLON, ROOT, SmallData, smalldata_path
from automask.io.read_xtc import JUNGFRAU_NAME, XTC_DIR, calib_dir, local_run_source
from automask.io.smalldata import Lcls1SmallDataDetectors
from automask.run_profile import RunProfile


_PAYLOAD_ACCESSOR_VETO = {
    "TypeId", "Version", "calib", "data", "frame", "image", "raw", "waveform"
}


def _payload_type_name(payload_type):
    if payload_type is None:
        return "—"
    module = payload_type.__module__
    module = module[len("psana."):] if module.startswith("psana.") else module
    return f"{module}.{payload_type.__name__}"


def configure_psana_environment():
    """Configure psana and optional SLAC tools in the current interpreter."""
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
    environment = {
        name: os.environ[name]
        for name in ("SIT_PSDM_DATA", "SIT_ROOT", "SIT_DATA")
    }

    checkout = os.environ.get("SMALLDATA_TOOLS") or config.get("SMALLDATA_TOOLS")
    if checkout:
        checkout = str(Path(checkout).expanduser().resolve())
        if not (Path(checkout) / "smalldata_tools").is_dir():
            raise RuntimeError(
                f"Invalid SMALLDATA_TOOLS checkout: {checkout}/smalldata_tools "
                "does not exist."
            )
        os.environ["SMALLDATA_TOOLS"] = checkout
        pythonpath = os.environ.get("PYTHONPATH", "").split(os.pathsep)
        if checkout not in pythonpath:
            os.environ["PYTHONPATH"] = os.pathsep.join(
                [checkout, *filter(None, pythonpath)]
            )
        if checkout not in sys.path:
            sys.path.insert(0, checkout)
        environment["SMALLDATA_TOOLS"] = checkout

    return environment


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


def _short_values(field, value):
    array = np.asarray(value)
    if array.ndim == 0:
        return {field: array.item()}
    if array.ndim != 1 or array.size > 64 or array.dtype.kind not in "biufcUS":
        raise ValueError("field is not a short numeric vector")
    return {
        f"{field}[{index}]": item.item() if hasattr(item, "item") else item
        for index, item in enumerate(array)
    }


def _discover_payload_values(payload):
    values = {}
    errors = 0
    for accessor in dir(payload):
        if accessor.startswith("_") or accessor in _PAYLOAD_ACCESSOR_VETO:
            continue
        method = getattr(payload, accessor)
        if not callable(method):
            continue
        try:
            values.update(_short_values(accessor, method()))
        except (AttributeError, IndexError, TypeError, ValueError):
            errors += 1
    return values, errors


def _source_detector_name(source_name):
    match = re.fullmatch(r"(?:BldInfo|DetInfo)\((.*)\)", source_name)
    return match.group(1) if match else source_name


def _record_values(columns, metadata, group, values, n_events, **field_info):
    for field, value in values.items():
        try:
            extracted = _short_values(field, value)
        except (TypeError, ValueError):
            continue
        for expanded_field, item in extracted.items():
            field_id = f"{group}/{expanded_field}"
            if field_id not in columns:
                columns[field_id] = [None] * n_events
                metadata[field_id] = {
                    **field_info,
                    "source": group,
                    "field": expanded_field,
                }
            columns[field_id][-1] = item


def _profile_array(values):
    present = [value for value in values if value is not None]
    numeric = present and all(
        isinstance(value, (bool, int, float, complex, np.number))
        for value in present
    )
    return np.asarray(
        [np.nan if value is None and numeric else value for value in values],
        dtype=float if numeric else object,
    )


def _column_summary(values):
    if values.dtype.kind not in "biufc":
        available_mask = np.asarray([value is not None for value in values])
        available = int(available_mask.sum())
        if not available:
            return "0.0%", "unavailable", "no values"
        text_values = np.asarray([
            value.decode("utf-8", errors="replace")
            if isinstance(value, bytes) else str(value)
            for value in values
        ], dtype=object)
        observed = text_values[available_mask]
        unique, counts = np.unique(observed, return_counts=True)
        coverage = f"{available / values.size:.1%}"
        if unique.size == 1:
            return coverage, "constant", unique[0]
        changes = int(np.count_nonzero(
            available_mask[1:] & available_mask[:-1]
            & (text_values[1:] != text_values[:-1])
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


def profile_run_values(
    run, source=None, detector_set=None, max_events=None, show=True
) -> RunProfile:
    """Profile one run through smalldata_tools and psana payload discovery."""
    source = source or local_run_source(run)
    data_source = source.open()
    detector_set = detector_set or Lcls1SmallDataDetectors(data_source)
    payloads = {}
    columns = {}
    field_metadata = {}
    evr_history = {}
    n_events = 0

    for event in data_source.events():
        n_events += 1
        for values in columns.values():
            values.append(None)
        for history in evr_history.values():
            history.append(None)

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
                "errors": 0,
            })
            info["events"] += 1
            if payload_type is None:
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

            if _source_detector_name(source_name) in detector_set.covered_sources:
                continue
            discovered, errors = _discover_payload_values(payload)
            info["errors"] += errors
            _record_values(
                columns,
                field_metadata,
                f"{source_name}/{type_name}",
                discovered,
                n_events,
                type=type_name,
                origin="psana payload",
            )

        for detector_name, detector_values in detector_set.read(event).items():
            if not isinstance(detector_values, dict):
                continue
            group = "EPICS" if detector_name == "epics" else detector_name
            _record_values(
                columns,
                field_metadata,
                group,
                detector_values,
                n_events,
                type="EPICS" if group == "EPICS" else "smalldata_tools",
                origin="smalldata_tools",
            )

        if max_events is not None and n_events >= max_events:
            break

    if not n_events:
        raise RuntimeError(f"Run {run:04d} contains no decodable events")

    for identity, history in evr_history.items():
        codes = sorted(set().union(*(codes or set() for codes in history)))
        source_name, type_name, _key_name = identity
        for code in codes:
            field = f"eventCode[{code}]"
            field_id = f"{source_name}/{type_name}/{field}"
            columns[field_id] = [
                None if event_codes is None else float(code in event_codes)
                for event_codes in history
            ]
            field_metadata[field_id] = {
                "source": source_name,
                "type": type_name,
                "field": field,
                "origin": "psana EVR",
            }

    value_arrays = {
        field_id: _profile_array(values)
        for field_id, values in columns.items()
    }
    epics_arrays = {
        field_id: values
        for field_id, values in value_arrays.items()
        if field_id.startswith("EPICS/")
    }
    payload_rows = [
        {
            "source": info["source"],
            "alias": info["alias"],
            "type": info["type"],
            "key": info["key"],
            "coverage": f'{info["events"] / n_events:.1%}',
            "read errors": info["errors"],
        }
        for info in sorted(
            payloads.values(),
            key=lambda row: (row["source"], row["type"], row["key"]),
        )
    ]

    value_rows = []
    for field_id, values in sorted(value_arrays.items()):
        if field_id.startswith("EPICS/"):
            continue
        coverage, variation, observed = _column_summary(values)
        metadata = field_metadata[field_id]
        value_rows.append({
            "source": metadata["source"],
            "field": metadata["field"],
            "type": metadata["type"],
            "origin": metadata["origin"],
            "coverage": coverage,
            "variation": variation,
            "observed": observed,
        })

    epics_rows = []
    for field_id, values in sorted(epics_arrays.items()):
        coverage, variation, observed = _column_summary(values)
        alias = field_id.removeprefix("EPICS/")
        epics_rows.append({
            "alias": alias,
            "PV": detector_set.epics_metadata.get(alias, alias),
            "dtype": str(values.dtype),
            "coverage": coverage,
            "variation": variation,
            "observed": observed,
        })

    if show:
        _show_table(f"Raw XTC payloads ({n_events:,} decoded events)", payload_rows)
        _show_table("Profiled per-shot values", value_rows)
        _show_table("EPICS process variables", epics_rows)
    return RunProfile(
        run=int(run),
        events=n_events,
        payloads=payload_rows,
        values=value_arrays,
        summary={"xtc": value_rows, "epics": epics_rows},
        epics=epics_rows,
        source=source,
    )


def print_detector_geometry(run, detector_name=JUNGFRAU_NAME, source=None):
    import psana

    source = source or local_run_source(run)
    data_source = source.open()
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
