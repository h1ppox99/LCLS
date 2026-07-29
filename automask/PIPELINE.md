# Setup

## Data

> [!info] Content of files
> 1. Collection of shots for each run (e.g. run 475 → 3201 shots, run 389 → 40 k+ shots)
> 2. Calibration data (e.g. one calibration every $\sim$ 70 run) 

## Current formalization

- *Input* : Tensor of shape `(n_shots, n_pixels)` (e.g. 800 shots, 4M pixels)
- *Output* : Tensor of shape `(n_pixels,)` (mask for each pixel)

> [!check] Current assumption
> **Static** masking : the mask is computed once for a given run and applied to all shots

## Systematic masking (detector dependent)


> [!info] Masks applied **deterministically** → independent of run
> 1. **Geometry** **mask** : computed directly from the arrangement of panels of the detector (filter for $0$ values)
> 2. **Calibration mask** : computed directly from the calibration data where `psana` defines the pixel status (`0` = good pixel / `1` hot rms · `2` cold rms · `4` saturated · `8` cold etc. = bad pixel)
> 
> *Post-processing* : padding (2+1) → pixels near edges are not trusted

| ![[Pasted image 20260728092519.png\|291]] | ![[Pasted image 20260728095727.png\|291]] |
| ----------------------------------------- | ----------------------------------------- |

# Agentic masking

The pipeline is decomposed into different parts, that each are critical to obtain a good mask :

1. **Shot selection**
2. **Reduction**
3. **Statistics**
4. **Regularization**
5. [Optional] Combination

![[Pasted image 20260728113114.png]]

## Shot selection 

**Output** : 3D tensor of shape `(n_shots, n_pixels_x, n_pixels_y)` 

```python
XRayClass = Literal["on", "off", "any"]
LaserClass = Literal["on", "off", "any"]
Normalization = Literal["none", "ipm2"]

class ShotSelection:
	xray: XRayClass = "on"
	laser: LaserClass = "off" # look at CC and VCC - threshold voltage
	n_shots: Optional[int] = 800
	filter_low: float = 0.03 # other ways of choosing
	filter_high: float = 0.03
	normalization: Normalization = "none"
	
	# Also accepts the calibration data as selection
```

See [[Shot selection]] for exploration of different shot selections.

## Reduction

**Input** : 3D tensor of shape `(n_shots, n_pixels_x, n_pixels_y)`
**Output** : 2D tensor of shape `(n_pixels_x, n_pixels_y)`

Reduction takes the selected shots and computes a single image from them. It **collapses the event axis.** 

```python
Reduction = Literal["mean", "std", "median", "mad"]
```

See [[Shot selection]] too for exploration of different reduction methods.

## Statistics 

**Input** : 2D tensor of shape `(n_pixels_x, n_pixels_y)`
**Output** : 2D tensor of shape `(n_pixels_x, n_pixels_y)` with either 
1. a continuous value (`field`) 
2. a boolean value (`pick`) 

A statistic consumes a **named feature** (`reduction ∘ ShotSelection`).

| name             | kind  | feature(s)        | default                                       | what it flags                                    |
| ---------------- | ----- | ----------------- | --------------------------------------------- | ------------------------------------------------ |
| `variance`       | field | `ustd`            | `k=3.5`, `mode="low"`                         | **low** dispersion = dead / shadowed / beam-stop |
| `sigma_clipping` | field | `umean`, `center` | `k=5`, `mode="both"`                          | pyFAI azimuthal sigma-clip residual z            |
| `asic_polish`    | field | `pedestal`        | `asic=256`, `n_iter=3`, `k=15`, `mode="high"` | pedestal defects `pixel_status` misses           |
| `hough_lines`    | pick  | `umean` (+ floor) | `width=1`, `line_length≈100–200`              | straight scratches / wire shadows / seams        |

## Regularization

**Input** : 2D tensor of shape `(n_pixels_x, n_pixels_y)` with either 
1. a continuous value (`field`) 
2. a boolean value (`pick`) 
**Output** : 2D tensor of shape `(n_pixels_x, n_pixels_y)` with either 
3. a continuous value (`field`) 
4. a boolean value (`pick`) 

### Continuous regularization

1. **Total variation** : `tv` $u=\arg\min_u \lVert u-z\rVert^2 + \text{weight}\cdot TV(u)$
See [[Slides meeting W4#TV method]]

2. **Blob scale** : `blob_scale`, `radii=(6, 9, 12, 16)` multi-scale matched filter. 
At radius $R$ the field is averaged over a flat disk and divided by $\sqrt{N_R}$, so unit-variance noise stays unit-variance (`k` stays in σ) while a coherent excess filling the disk is amplified by $\sqrt{N_R}$

$\approx$ similar to kernel regression

### Boolean regularization

1. **Pad** : `pad` (dilate)
2. **Fill holes** : `fill_holes`
3. **Area gate** : `area_gate` (drop small components)

## [Optional] Combination

When different shots and different statistics are used, the resulting masks can be combined in different ways:

- `union` (**default**) 

Different directions have been explored but require to use only field statistics (`weighted_sum`,  `mahalanobis`) and do not provide much improvement. If each mask corresponds to a different source of noise, then the union is the most interpretable and robust way to combine them. 

# Evaluation

## Current setup

### Metrics

1. IoU / precision / recall vs the human masks on `EVAL_RUNS = [389, 475]`
*Issue* : only **2** ground truth masks available + masks are not perfect

2. Visual inspection 
*Issue* : still only two runs available + subjective judgement

3. Inspection of integrated 1D profile 
*Issue* : not metric to evaluate it → only physical knowledge of Taekeun can help

### Synthetic injection of artifacts

Same as presented in [[Slides meeting W4#Evaluation]] : inject artifacts on clean images so that ground truth mask is known.  

## Results

#### Run 389

![[panels_run0389.png]]

![[azimuthal_run0389.png]]

#### Run 475


![[panels_run0475.png]]

![[azimuthal_run0475.png]]



# Next steps


## Data

No reliable progress can be made using only these two runs. Most of the masking work is done by removing the rectangular shadow. All the remaining improvements are small and not significant in terms of overall performance of the system. Robustness and performance on other runs would be much more significant. 
## Masking

- Formalize better difference between each technique 
→ kernel regression allows for efficient recall
→ non-linear variational methods allow for refined precision

- Robustness of combination of techniques + hyperparameter choosing
→ decide how unsupervised / supervised by the agent/evaluation the method should be

## Evaluation

- **Internal consistency** of radial average across azimuthal sections : currently investigating comparisons between masks (including random controls), robustness to noise, and sensitivity to hyperparameters
- Build strong statistical tests to evaluate the quality of the mask

# Follow-up work

- [ ] Check contents of XTC files and different entries
- [ ] Add new data to repo and run evaluation on these runs
- [ ] Formalization of different techniques
- [ ] Study of evaluation metrics to build → statistically grounded