import pyarrow.dataset as ds
import treecorr as tc
from query_parquet_helpers import check

# CONFIG
CATALOG = "../Data.nosync/redmagic_parquet/redmagic_small_processed.parquet"
OUTPUT = "small_patch_centers.txt"

# Perform some simple sanity checks
check(files=[CATALOG])

dataset = ds.dataset(
    CATALOG,
    format="parquet",
)

centroids_cat = tc.Catalog(
    ra=dataset.to_table().combine_chunks()["ra"],
    dec=dataset.to_table().combine_chunks()["dec"],
    ra_units="deg",
    dec_units="deg",
    npatch=5,
)

centroids_cat.write_patch_centers(OUTPUT)
