# XPP CO2 Data Processing

This repository contains a notebook-based workflow for loading XPP/LCLS data, calibrating a detector mask and `q` map, and converting Jungfrau detector images into delay-binned 2D and 1D scattering outputs.

The code is written around experiment `xppl1016922` and uses `psana` together with smalldata HDF5 files on the LCLS filesystem.

## What Takes Human Time

The most time-consuming part of this workflow is not the event loop itself. The main human effort is:

- finding a reliable detector mask
- choosing a stable normalization strategy from X-ray intensity diagnostics
- deciding which shots to keep based on `ipm2`, `d1`, and related diagnostics
- checking whether the resulting 2D sums and 1D radial averages look physically reasonable

In practice, the mask and the normalization are tuned together. A typical iteration is:

1. Inspect intensity diagnostics and summed images.
2. Reject bad shots or unstable intensity regions.
3. Adjust the detector mask and beam-center choice.
4. Re-run the radial averaging and verify that the curves are consistent.

## Repository Contents

- `1_Loaddata.ipynb`: loads summary diagnostics and example detector images.
- `2_Lab6_Mask_calibration.ipynb`: builds the detector mask, sets the beam center, and generates `qmap.npy`.
- `3_Demo_CO2dataprocessing_share.ipynb`: demonstrates delay-binned processing and plots saved 2D and 1D outputs.
- `utils.py`: helper functions for summary loading, delay binning, saving outputs, and radial averaging.

## Requirements

This workflow is intended to run in an environment with access to LCLS data and `psana`.

You also need access to smalldata HDF5 files under:

- `/sdf/data/lcls/ds/xpp/xppl1016922/`

## Calibration Files

The workflow generates the following local calibration files:

- `Mask.npy`: boolean detector mask created in the calibration notebook
- `qmap.npy`: pixel-wise scattering vector map
- `correction.npz`: polarization and solid-angle correction maps generated in the calibration notebook

At the moment, `utils.py` uses `Mask.npy` and `qmap.npy` directly during processing.

## Recommended Workflow

1. Open `1_Loaddata.ipynb`.
   - Inspect `ipm2`, `d1`, `d5`, `d6`, delay values, and summed detector images.
   - Use this step to understand shot-to-shot intensity stability before committing to cuts.

2. Run `2_Lab6_Mask_calibration.ipynb`.
   - Build the zero-value mask from the summed image.
   - Add the manual geometry mask for detector regions that should be excluded.
   - Set the beam center and ring geometry used to generate `qmap.npy`.
   - Save `Mask.npy`.

3. Run `3_Demo_CO2dataprocessing_share.ipynb`.
   - Process detector images into delay bins.
   - Save normalized 2D images and 1D radial averages.
   - Plot selected delays and runs from the saved outputs.

## Processing Logic

The processing code in `utils.py` uses the following logic:

- `d1` is taken from `diode2/channels[:,0]`.
- The `VCCC` selection keeps events with `vcc > 4`.
- Events are also filtered by the selected `d1` range.
- Each accepted detector image is normalized by the sample-diode signal before accumulation.
- Negative pixel values are clipped to zero after normalization.
- Each event is assigned to the nearest target delay bin.
- The code processes both `VCCC` and `CC` selections and saves each result separately.

## Helper Functions

`utils.py` provides four main functions:

- `get_summary(runs)`: loads diode, intensity-monitor, delay, and selection diagnostics from smalldata files.
- `extract_1d_data_from_q_mapping(sumimg, TotalMask, qmap, num_iso_q_lines=500)`: converts a 2D image into a 1D radial average.
- `save_delay_data(...)`: saves delay-binned 2D and 1D outputs.
- `process_run_and_save(...)`: processes one run for both `VCCC` and `CC` selections and writes results to disk.

## Output Files

Processed outputs are written to `Processeddata/` and include files such as:

- `VCCC_sum_images_by_delay_run<run>.npy`
- `VCCC_img_index_by_delay_run<run>.npy`
- `VCCC_normalized_images_by_delay_run<run>.npy`
- `VCCC_average_values_run<run>.npy`
- `VCCC_counts_by_delay_run<run>.npy`
- `VCCC_iso_q_lines_run<run>.npy`
- `VCCC_q_centers_run<run>.npy`
- corresponding `CC_*` files for the complementary selection

The saved dictionaries are keyed by the target delay values used during processing.

## Notes

- The calibration notebook currently uses manual mask regions and a manually chosen beam center.
- The best mask and normalization settings may need to be re-tuned for a different run range.
- The demo notebook is an example workflow and can be expanded for production processing over more runs and more images.
