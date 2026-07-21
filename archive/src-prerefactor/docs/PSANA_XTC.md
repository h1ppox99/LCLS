## 5. Extract frames — `io/read_xtc.py`

```bash
source psana_env.sh                    # run from src/
python io/read_xtc.py --run 475 --events 200
# writes automask/outputs/cache/xtc_run0475_frames.h5:
#   frames    (200, 2, 512, 1024)   calibrated
#   sum_frame (2, 512, 1024)        running sum