#!/usr/bin/env python3
\"\"\"Build the 1,000-image public diffraction corpus in ``<repo>/data``.

The downloader is deliberately separate from normalization.  Binary source
archives are retained in ``data/.downloads`` and every derived ``.npy`` image
is traceable through ``data/manifest.jsonl``.
\"\"\"
from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import tarfile
import urllib.request
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / \"data\"
DOWNLOADS = DATA / \".downloads\"
IMAGES = DATA / \"images\"
SEED = 20260718

SOURCES = {
    \"cxidb009\": {
        \"cxidb_id\": 9,
        \"doi\": \"10.11577/1096911\",
        \"title\": \"Cryptotomography: reconstructing 3D Fourier intensities from randomly oriented single-shot diffraction patterns\",
        \"facility\": \"FLASH\",
        \"sample\": \"iron oxide ellipsoids\",
        \"license\": \"CC0-1.0\",
        \"url\": \"https://www.cxidb.org/data/9/cxidb-9.tar.gz\",
        \"archive\": \"cxidb-9.tar.gz\",
        \"quota\": 400,
    },
    \"cxidb013\": {
        \"cxidb_id\": 13,
        \"doi\": \"10.11577/1096915\",
        \"title\": \"Femtosecond free-electron laser x-ray diffraction data sets for algorithm development\",
        \"facility\": \"LCLS\",
        \"sample\": \"T4 bacteriophage\",
        \"license\": \"CC0-1.0\",
        \"url\": \"https://www.cxidb.org/data/13/cxidb-13-amo10510-r0162.tar\",
        \"archive\": \"cxidb-13-r0162.tar\",
        \"quota\": 70,
    },
    \"cxidb016\": {
        \"cxidb_id\": 16,
        \"doi\": \"10.11577/1096919\",
        \"title\": \"Fractal morphology, imaging and mass spectrometry of single aerosol particles in flight\",
        \"facility\": \"LCLS\",
        \"sample\": \"aerosols and soot\",
        \"license\": \"CC0-1.0\",
        \"url\": \"https://www.cxidb.org/data/16/cxidb-16.tar.gz\",
        \"archive\": \"cxidb-16.tar.gz\",
        \"quota\": 200,
    },
    \"cxidb020\": {
        \"cxidb_id\": 20,
        \"doi\": \"10.11577/1096925\",
        \"title\": \"Single-particle structure determination by correlations of snapshot X-ray diffraction patterns\",
        \"facility\": \"LCLS\",
        \"sample\": \"polystyrene-sphere clusters\",
        \"license\": \"CC0-1.0\",
        \"url\": \"https://www.cxidb.org/data/20/cxidb-20.tar.gz\",
        \"archive\": \"cxidb-20.tar.gz\",
        \"quota\": 250,
    },
    \"cxidb026\": {
        \"cxidb_id\": 26,
        \"doi\": \"10.11577/1169686\",
        \"title\": \"Imaging single cells in a beam of live cyanobacteria with an X-ray laser\",
        \"facility\": \"LCLS\",
        \"sample\": \"Cyanobium gracile\",
        \"license\": \"CC0-1.0\",
        \"url\": \"https://www.cxidb.org/data/26/cxidb-26.cxi\",
        \"archive\": \"cxidb-26.cxi\",
        \"quota\": 10,
    },
    \"cxidb057\": {
        \"cxidb_id\": 57,
        \"doi\": \"10.11577/1345570\",
        \"title\": \"Diffraction data of core-shell nanoparticles from X-ray Free Electron Laser\",
        \"facility\": \"LCLS\",
        \"sample\": \"gold-core/palladium-shell nanoparticles\",
        \"license\": \"CC0-1.0\",
        \"url\": \"https://www.cxidb.org/data/57/normal-incidence-pattern.cxi\",
        \"archive\": \"cxidb-57.cxi\",
        \"quota\": 70,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open(\"rb\") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b\"\"):
            digest.update(block)
    return digest.hexdigest()


def download_sources() -> None:
    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    for source in SOURCES.values():
        target = DOWNLOADS / source[\"archive\"]
        if target.exists():
            print(f\"already downloaded: {target.name}\")
            continue
        partial = target.with_suffix(target.suffix + \".part\")
        print(f\"downloading {source['url']}\")
        try:
            urllib.request.urlretrieve(source[\"url\"], partial)
            partial.replace(target)
        except BaseException:
            print(f\"partial download retained at {partial}\")
            raise


def cxi_bytes(member: tarfile.TarInfo, archive: tarfile.TarFile) -> bytes:
    stream = archive.extractfile(member)
    if stream is None:
        raise OSError(f\"cannot read {member.name}\")
    return stream.read()


def detector_pair(h5: h5py.File, first: int, second: int) -> np.ndarray:
    base = \"entry_1/instrument_1\"
    top = h5[f\"{base}/detector_{first}/data\"][()]
    bottom = h5[f\"{base}/detector_{second}/data\"][()]
    return np.vstack((top, bottom))


def smallest_lossless_integer(array: np.ndarray) -> np.ndarray:
    \"\"\"Reduce legacy int64 detector frames without changing a single value.\"\"\"
    if array.dtype.kind not in \"iu\" or array.size == 0:
        return array
    low, high = int(array.min()), int(array.max())
    candidates = (np.int8, np.int16, np.int32, np.int64) if low < 0 else (
        np.uint8, np.uint16, np.uint32, np.uint64
    )
    for dtype in candidates:
        limits = np.iinfo(dtype)
        if limits.min <= low and high <= limits.max:
            return array.astype(dtype, copy=False)
    return array


def write_image(
    source_key: str,
    source_item: str,
    image: np.ndarray,
    index: int,
    *,
    assembly: str,
    category: str = \"\",
) -> dict:
    if image.ndim != 2:
        raise ValueError(f\"{source_item}: expected a 2-D image, got {image.shape}\")
    original_dtype = str(image.dtype)
    image = smallest_lossless_integer(np.asarray(image))
    filename = f\"{source_key}_{index:04d}.npy\"
    target = IMAGES / filename
    np.save(target, image, allow_pickle=False)
    info = SOURCES[source_key]
    return {
        \"id\": target.stem,
        \"path\": str(target.relative_to(ROOT)),
        \"source\": source_key,
        \"cxidb_id\": info[\"cxidb_id\"],
        \"source_item\": source_item,
        \"category\": category,
        \"facility\": info[\"facility\"],
        \"sample\": info[\"sample\"],
        \"shape\": list(image.shape),
        \"original_dtype\": original_dtype,
        \"stored_dtype\": str(image.dtype),
        \"assembly\": assembly,
        \"minimum\": float(np.nanmin(image)),
        \"maximum\": float(np.nanmax(image)),
        \"sha256\": sha256(target),
    }


def tar_members(path: Path) -> list[tarfile.TarInfo]:
    with tarfile.open(path) as archive:
        return [m for m in archive.getmembers() if m.isfile() and m.name.endswith(\".cxi\")]


def choose_cxidb9(path: Path, quota: int, rng: random.Random) -> list[str]:
    \"\"\"Mix high-signal shots and random shots (including useful blanks).\"\"\"
    ranked: list[tuple[int, str]] = []
    with tarfile.open(path) as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(\".cxi\"):
                continue
            with h5py.File(io.BytesIO(cxi_bytes(member, archive)), \"r\") as h5:
                total = int(h5[\"entry_1/instrument_1/detector_1/data_sum\"][()])
            ranked.append((total, member.name))
    ranked.sort(reverse=True)
    high_signal = [name for _, name in ranked[: quota // 2]]
    remaining = [name for _, name in ranked[quota // 2 :]]
    return sorted(high_signal + rng.sample(remaining, quota - len(high_signal)))


def build_tar_source(
    source_key: str,
    selected_names: list[str],
    pair: tuple[int, int] | None,
    records: list[dict],
) -> None:
    path = DOWNLOADS / SOURCES[source_key][\"archive\"]
    selected = set(selected_names)
    index = 0
    with tarfile.open(path) as archive:
        for member in archive:
            if member.name not in selected:
                continue
            with h5py.File(io.BytesIO(cxi_bytes(member, archive)), \"r\") as h5:
                if pair is None:
                    image = h5[\"entry_1/instrument_1/detector_1/data\"][()]
                    assembly = \"native 2-D detector image\"
                else:
                    image = detector_pair(h5, *pair)
                    assembly = f\"vertical stack of CXI detector_{pair[0]} (top) and detector_{pair[1]} (bottom)\"
            parts = Path(member.name).parts
            category = parts[-2] if source_key == \"cxidb016\" else \"\"
            records.append(write_image(source_key, member.name, image, index, assembly=assembly, category=category))
            index += 1
    if index != len(selected):
        raise RuntimeError(f\"{source_key}: found {index}/{len(selected)} selected images\")


def build_corpus() -> None:
    missing = [s[\"archive\"] for s in SOURCES.values() if not (DOWNLOADS / s[\"archive\"]).exists()]
    if missing:
        raise FileNotFoundError(f\"missing source downloads: {missing}; run with --download\")
    IMAGES.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    records: list[dict] = []

    path9 = DOWNLOADS / SOURCES[\"cxidb009\"][\"archive\"]
    build_tar_source(\"cxidb009\", choose_cxidb9(path9, 400, rng), None, records)

    members13 = [m.name for m in tar_members(DOWNLOADS / SOURCES[\"cxidb013\"][\"archive\"])]
    build_tar_source(\"cxidb013\", sorted(rng.sample(members13, 70)), (3, 4), records)

    members16 = tar_members(DOWNLOADS / SOURCES[\"cxidb016\"][\"archive\"])
    special16 = [m.name for m in members16 if \"/phased/\" in m.name or \"/soot/\" in m.name]
    tof16 = [m.name for m in members16 if \"/tof/\" in m.name]
    selected16 = special16 + rng.sample(tof16, 200 - len(special16))
    build_tar_source(\"cxidb016\", sorted(selected16), (1, 2), records)

    members20 = [m.name for m in tar_members(DOWNLOADS / SOURCES[\"cxidb020\"][\"archive\"])]
    build_tar_source(\"cxidb020\", sorted(rng.sample(members20, 250)), (1, 2), records)

    path26 = DOWNLOADS / SOURCES[\"cxidb026\"][\"archive\"]
    with h5py.File(path26, \"r\") as h5:
        for index in range(10):
            key = f\"entry_{index + 1}/instrument_1/detector_1/data\"
            records.append(write_image(\"cxidb026\", key, h5[key][()], index, assembly=\"native 2-D detector image\"))

    path57 = DOWNLOADS / SOURCES[\"cxidb057\"][\"archive\"]
    with h5py.File(path57, \"r\") as h5:
        dataset = h5[\"entry_1/instrument_1/detector_1/data\"]
        chosen = sorted(rng.sample(range(dataset.shape[0]), 70))
        for index, frame_index in enumerate(chosen):
            records.append(write_image(
                \"cxidb057\", f\"entry_1/instrument_1/detector_1/data[{frame_index}]\",
                dataset[frame_index], index, assembly=\"native 2-D detector image\"
            ))

    if len(records) != 1000:
        raise RuntimeError(f\"expected 1000 images, built {len(records)}\")
    manifest = DATA / \"manifest.jsonl\"
    with manifest.open(\"w\", encoding=\"utf-8\") as stream:
        for record in sorted(records, key=lambda item: item[\"id\"]):
            stream.write(json.dumps(record, sort_keys=True) + \"\
    source_manifest = {
        key: {**value, \"archive_sha256\": sha256(DOWNLOADS / value[\"archive\"])}
        for key, value in SOURCES.items()
    }
    (DATA / \"sources.json\").write_text(json.dumps(source_manifest, indent=2) + \"\
    print(f\"built {len(records)} images; manifest: {manifest}\")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(\"--download\", action=\"store_true\", help=\"download missing CC0 source archives\")
    parser.add_argument(\"--download-only\", action=\"store_true\", help=\"download but do not normalize\")
    args = parser.parse_args()
    if args.download or args.download_only:
        download_sources()
    if not args.download_only:
        build_corpus()


if __name__ == \"__main__\":
    main()