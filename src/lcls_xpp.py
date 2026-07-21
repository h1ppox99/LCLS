# ---- paths -----------------------------------------------------------------
# Root of the local data copy (the dir that contains xtc/, hdf5/, calib/ ...).
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/io/ -> LCLS
COLON = chr(0xF022)          # the char that stands in for ":" on disk
