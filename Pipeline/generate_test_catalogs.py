from query_parquet_helpers import subselect_parquet, check
import pyarrow.compute as pc
import os

# Config
MAIN_REDMAGIC_FILE = "../Data.nosync/redmagic_parquet/redmagic_full_processed.parquet"
MAIN_METACAL_FILE = "../Data.nosync/metacal_parquet/metacal_full_catalog.parquet"
REDMAGIC_OUTPUT_DIR = "../Data.nosync/redmagic_parquet"
REDMAGIC_OUTPUT_NAME = "redmagic_5deg"  # "".parquet" at the end is implied
METACAL_OUTPUT_DIR = "../Data.nosync/metacal_parquet"
METACAL_OUTPUT_NAME = "metacal_5deg"  # "".parquet" at the end is implied

RA_MAX = 40
RA_MIN = 35
DEC_MAX = -35
DEC_MIN = -40

# Perform some simple sanity checks
check(
    files=[MAIN_REDMAGIC_FILE, MAIN_METACAL_FILE],
    dirs=[REDMAGIC_OUTPUT_DIR, METACAL_OUTPUT_DIR],
)

# Write the files
position_filter = (
    (pc.field("ra") < RA_MAX)
    & (pc.field("ra") > RA_MIN)
    & (pc.field("dec") < DEC_MAX)
    & (pc.field("dec") > DEC_MIN)
)

print("Writing redmagic file.")
subselect_parquet(
    MAIN_REDMAGIC_FILE,
    position_filter,
    REDMAGIC_OUTPUT_DIR,
    REDMAGIC_OUTPUT_NAME + "{0}.parquet",
)

print("Writing metacal file.")
subselect_parquet(
    MAIN_METACAL_FILE,
    position_filter,
    METACAL_OUTPUT_DIR,
    METACAL_OUTPUT_NAME + "{0}.parquet",
)

os.rename(
    REDMAGIC_OUTPUT_DIR + "/" + f"{REDMAGIC_OUTPUT_NAME}0.parquet",
    REDMAGIC_OUTPUT_DIR + "/" + f"{REDMAGIC_OUTPUT_NAME}.parquet",
)
os.rename(
    METACAL_OUTPUT_DIR + "/" + f"{METACAL_OUTPUT_NAME}0.parquet",
    METACAL_OUTPUT_DIR + "/" + f"{METACAL_OUTPUT_NAME}.parquet",
)
