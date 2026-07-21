
COLON = chr(0xF022)                       # the on-disk stand-in for ':'
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # .../LCLS (src/io/ -> LCLS)
INSTRUMENT = "xpp"
EXPERIMENT = "xppl1016922"
# Where to build the psana-style tree.  On /Data (lots of free space), NOT home.
PSDM = os.environ.get("SIT_PSDM_DATA",
                      os.path.join(os.path.dirname(ROOT), "psdm"))