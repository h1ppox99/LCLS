import os
import time
import numpy as np
from psana import MPIDataSource, Detector
import h5py


def extract_1d_data_from_q_mapping(sumimg, TotalMask, qmap, num_iso_q_lines=500):
    """
    Compute a 1D radial average from a 2D detector image using a q-map.

    Parameters
    ----------
    sumimg : np.ndarray
        2D detector image.
    TotalMask : np.ndarray
        Boolean mask array with the same shape as `sumimg`.
        True indicates masked pixels that should be excluded.
    qmap : np.ndarray
        2D q-value map with the same shape as `sumimg`.
    num_iso_q_lines : int, optional
        Number of q-bin edges. The resulting 1D curve has
        `num_iso_q_lines - 1` points.

    Returns
    -------
    iso_q_lines : np.ndarray
        Q-bin edges.
    average_values : np.ndarray
        Mean intensity in each q-bin.
    """
    assert qmap.shape == sumimg.shape == TotalMask.shape, \
        "qmap, sumimg, and TotalMask must have the same shape"

    # Use slightly less than the full q-range to avoid edge artifacts.
    q_max = np.max(qmap) * 0.99
    iso_q_lines = np.linspace(0, q_max, num_iso_q_lines)

    average_values = []
    for i in range(len(iso_q_lines) - 1):
        q_min = iso_q_lines[i]
        q_hi = iso_q_lines[i + 1]

        # Select pixels inside the current q-bin and not masked.
        mask = (qmap >= q_min) & (qmap < q_hi) & (~TotalMask)

        if np.any(mask):
            average_values.append(np.mean(sumimg[mask]))
        else:
            average_values.append(np.nan)

    return iso_q_lines, np.array(average_values)


def save_delay_data(
    run_num,
    sum_images_by_delay,
    img_index_by_delay,
    prefix,
    qmap,
    TotalMask,
    outdir="Processeddata",
    save_normalized_2d=True,
    num_iso_q_lines=500,
):
    """
    Save processed outputs for one run and one data category (e.g. VCCC or CC).

    Saved outputs
    -------------
    1. Summed 2D images grouped by delay
    2. Event indices used for each delay
    3. Shot-count-normalized 2D images (optional)
    4. 1D radial averages for each delay
    5. Q-bin edges and Q-bin centers
    6. Number of selected shots for each delay
    """
    os.makedirs(outdir, exist_ok=True)

    # Save the raw summed images and contributing event indices.
    np.save(f"{outdir}/{prefix}_sum_images_by_delay_run{run_num}.npy", sum_images_by_delay)
    np.save(f"{outdir}/{prefix}_img_index_by_delay_run{run_num}.npy", img_index_by_delay)

    normalized_images_by_delay = {}
    average_values_by_delay = {}
    counts_by_delay = {}

    iso_q_lines_saved = None

    for delay, sum_img in sum_images_by_delay.items():
        nshots = len(img_index_by_delay[delay])
        counts_by_delay[delay] = nshots

        # If no valid images contributed to this delay bin, store None.
        if nshots == 0 or sum_img is None or np.sum(sum_img) == 0:
            normalized_images_by_delay[delay] = None
            average_values_by_delay[delay] = None
            continue

        # Normalize by the number of selected shots in this delay bin.
        normalized_img = sum_img / nshots
        normalized_images_by_delay[delay] = normalized_img

        # Convert the normalized 2D image into a 1D radial average.
        iso_q_lines, average_values = extract_1d_data_from_q_mapping(
            normalized_img,
            TotalMask,
            qmap,
            num_iso_q_lines=num_iso_q_lines,
        )
        average_values_by_delay[delay] = average_values

        # Save the q-axis only once since it is common to all delays.
        if iso_q_lines_saved is None:
            iso_q_lines_saved = iso_q_lines

    if save_normalized_2d:
        np.save(
            f"{outdir}/{prefix}_normalized_images_by_delay_run{run_num}.npy",
            normalized_images_by_delay,
        )

    np.save(f"{outdir}/{prefix}_average_values_run{run_num}.npy", average_values_by_delay)
    np.save(f"{outdir}/{prefix}_counts_by_delay_run{run_num}.npy", counts_by_delay)

    if iso_q_lines_saved is not None:
        np.save(f"{outdir}/{prefix}_iso_q_lines_run{run_num}.npy", iso_q_lines_saved)
        q_centers = 0.5 * (iso_q_lines_saved[:-1] + iso_q_lines_saved[1:])
        np.save(f"{outdir}/{prefix}_q_centers_run{run_num}.npy", q_centers)

    print(f"Saved {prefix} outputs for run {run_num}")


def get_summary(runs):
    """
    Read smalldata HDF5 files and concatenate summary diagnostics over runs.

    Parameters
    ----------
    runs : list of int
        Run numbers to load.

    Returns
    -------
    d1_all, d5_all, d6_all, sample_diode_all : np.ndarray
        Selected diode / channels used for event filtering and normalization.
    ipm2_all : np.ndarray
        IPM2 / Intensity monitor channels used for event filtering.
    delay_all : np.ndarray
        ps Delay.
    vcc_all : np.ndarray
        VCC shutter monitor channel.
    cc_all : np.ndarray
        CC shutter monitor channel.
    run_indices : list of int
        Starting global event index for each run in the concatenated arrays.
    """
    exp = 'xppl1016922'

    d1_all = []
    d5_all = []
    d6_all = []
    ipm2_all = []
    sample_diode_all = []
    delay_all = []
    vcc_all = []
    cc_all = []
    run_indices = []
    current_index = 0

    for run in runs:
        filepath = f'/sdf/data/lcls/ds/xpp/{exp}/hdf5/smalldata/{exp}_Run{run:04d}.h5'

        with h5py.File(filepath, 'r') as f:
            ipm2 = f['ipm2/sum'][()]
            cc = np.array(f['ai/ch02'])   # CC monitor; large value typically indicates open
            vcc = np.array(f['ai/ch03'])  # VCC monitor
            delay = f['epicsAll/delay'][()]

            # Collect only the channels that are used later in filtering / normalization.
            diodes = np.zeros((ipm2.size, 7))
            diodes[:, 1] = f['diode2/channels'][:, 0]
            diodes[:, 5] = f['diodeU/channels'][:, 3]
            diodes[:, 6] = f['lombpm/channels'][:, 2]
            sample_diode = f['diodeU/channels'][:, 0]

        # Record the starting index of this run in the concatenated arrays.
        run_indices.append(current_index)

        # Convert the delay readback to ps using the experiment calibration.
        delay_in_ps = 0.9376321852982434 * (delay - 3.1)

        d1_all = np.concatenate((d1_all, diodes[:, 1]))
        d5_all = np.concatenate((d5_all, diodes[:, 5]))
        d6_all = np.concatenate((d6_all, diodes[:, 6]))
        ipm2_all = np.concatenate((ipm2_all, ipm2))
        sample_diode_all = np.concatenate((sample_diode_all, sample_diode))
        delay_all = np.concatenate((delay_all, delay_in_ps))
        vcc_all = np.concatenate((vcc_all, vcc))
        cc_all = np.concatenate((cc_all, cc))

        current_index += len(ipm2)

    return d1_all, d5_all, d6_all, ipm2_all, sample_diode_all, delay_all, vcc_all, cc_all, run_indices


def process_run_and_save(
    run_num,
    target_delays,
    exp_name="xppl1016922",
    detname="jungfrau1M_alcove",
    qmap_path="qmap.npy",
    mask_path="Mask.npy",
    outdir="Processeddata",
    nimg=5,
    image_shape=(1064, 1030),
):
    """
    Process one run and save both VCCC and CC datasets.

    Workflow
    --------
    1. Read summary data from the smalldata file
    2. Select events using diagnostic-based masks
    3. Accumulate detector images into delay bins
    4. Save 2D and 1D outputs for both VCCC and CC selections
    """
    run_start_time = time.time()

    d1, d5, d6, ipm2, sample_diode, delay_in_ps, vcc, cc, run_indices = get_summary([run_num])

    qmap = np.load(qmap_path)
    TotalMask = np.load(mask_path)

    # =========================
    # VCCC selection
    # =========================
    ds = MPIDataSource(f"exp={exp_name}:run={run_num}:smd")
    det = Detector(detname)

    sum_images_by_delay = {delay: np.zeros(image_shape) for delay in target_delays}
    img_index_by_delay = {delay: [] for delay in target_delays}

    # Event rejection mask for VCCC.
    # Keep only events with:
    # - VCC open
    # - d1 in the selected range
    # - finite, nonzero sample diode for normalization
    mask_excluded = np.zeros(len(ipm2), dtype=bool)
    mask_excluded |= (vcc < 2)
    mask_excluded |= ~((d1 >= 0.4) & (d1 <= 0.45))
    mask_excluded |= ~np.isfinite(sample_diode)
    mask_excluded |= (sample_diode == 0)

    for nevt, evt in enumerate(ds.events()):
        if nevt >= len(mask_excluded) or mask_excluded[nevt]:
            continue

        img = det.image(evt)
        if img is None:
            continue

        # Assign the event to the nearest target delay.
        delay = delay_in_ps[nevt]
        closest_delay = min(target_delays, key=lambda x: abs(x - delay))

        # Normalize each image by the sample diode signal.
        tmp = np.copy(img) / sample_diode[nevt]

        # Remove negative values after normalization.
        tmp[tmp < 0] = 0

        sum_images_by_delay[closest_delay] += tmp
        img_index_by_delay[closest_delay].append(nevt)

        # Stop after collecting the requested number of accepted events.
        if sum(len(v) for v in img_index_by_delay.values()) >= nimg:
            break

    save_delay_data(
        run_num=run_num,
        sum_images_by_delay=sum_images_by_delay,
        img_index_by_delay=img_index_by_delay,
        prefix="VCCC",
        qmap=qmap,
        TotalMask=TotalMask,
        outdir=outdir,
    )

    print(f"Finished VCCC for run {run_num} in {time.time() - run_start_time:.2f} s")

    # =========================
    # CC selection
    # =========================
    run_start_time = time.time()

    # Re-create the datasource because ds.events() is consumed after one pass.
    ds = MPIDataSource(f"exp={exp_name}:run={run_num}:smd")
    det = Detector(detname)

    cc_sum_images_by_delay = {delay: np.zeros(image_shape) for delay in target_delays}
    cc_img_index_by_delay = {delay: [] for delay in target_delays}

    # Event rejection mask for CC.
    # Here we select the complementary condition relative to the VCCC selection.
    mask_excluded = np.zeros(len(ipm2), dtype=bool)
    mask_excluded |= (vcc > 2)
    mask_excluded |= ~((d1 >= 0.4) & (d1 <= 0.45))
    mask_excluded |= ~np.isfinite(sample_diode)
    mask_excluded |= (sample_diode == 0)

    for nevt, evt in enumerate(ds.events()):
        if nevt >= len(mask_excluded) or mask_excluded[nevt]:
            continue

        img = det.image(evt)
        if img is None:
            continue

        delay = delay_in_ps[nevt]
        closest_delay = min(target_delays, key=lambda x: abs(x - delay))

        tmp = np.copy(img) / sample_diode[nevt]
        tmp[tmp < 0] = 0

        cc_sum_images_by_delay[closest_delay] += tmp
        cc_img_index_by_delay[closest_delay].append(nevt)

        if sum(len(v) for v in cc_img_index_by_delay.values()) >= nimg:
            break

    save_delay_data(
        run_num=run_num,
        sum_images_by_delay=cc_sum_images_by_delay,
        img_index_by_delay=cc_img_index_by_delay,
        prefix="CC",
        qmap=qmap,
        TotalMask=TotalMask,
        outdir=outdir,
    )

    print(f"Finished CC for run {run_num} in {time.time() - run_start_time:.2f} s")