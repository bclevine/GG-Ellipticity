import os
import numpy as np
import gc

import pyarrow.dataset as ds
import pyarrow.compute as pc
import dask.dataframe as dd
import pyccl as ccl
import pandas as pd

# Config
SOURCE_FILE_TO_QUERY = "sources.parquet"
LENS_FILE_TO_QUERY = "lenses.parquet"
OUTPUT_DIR = "csv_output_folder"
REDSHIFT_BUFFER = 0.10

# Prepare the cosmology
cosmo = ccl.CosmologyVanillaLCDM()

# Define the redshift bins
z_min = 0.15
z_max = 0.90
z_edges = np.linspace(z_min, z_max, 31)


# Important for later:
def compute_median_redshift(file_to_query, filter_expression, column_name="Z"):
    dataset = ds.dataset(
        file_to_query,
        format="parquet",
    )
    dataset_filtered = dataset.filter(filter_expression).to_table()
    return np.nanmedian(dataset_filtered[column_name]), len(dataset_filtered)


def sigma_crit_helper(z_S, z_L, cosmo):
    # Returns the inverse of sigma_crit in units of pc^2 / Msun (proper distances)
    return 1 / (
        cosmo.sigma_critical(a_lens=1 / (1 + z_L), a_source=1 / (1 + z_S)) / 1e12
    )


# Main code
# For testing, we can just set the range to 1
# Slices the lens catalog into 30 narrow redshift bins and precomputes the weights
# for each source bin (with z_source > z_lens + z_buffer))
# Then save to a CSV file which is compatible with Treecorr.
for i in range(30):
    print(f"Processing file {i+1}/30.", flush=True)

    # Select all objects with z > z_max + z_buffer
    filter_exp = [
        ("photo_z", ">", z_edges[i + 1] + REDSHIFT_BUFFER),
        ("zmc", ">", z_edges[i + 1]),
    ]
    df = dd.read_parquet(SOURCE_FILE_TO_QUERY, filters=filter_exp)
    lens_filter_exp = (pc.field("Z") > z_edges[i]) & (pc.field("Z") < z_edges[i + 1])
    z_L, len_lens = compute_median_redshift(LENS_FILE_TO_QUERY, lens_filter_exp)
    if len_lens < 5:
        print(f"Length of lens sample = {len_lens}.", flush=True)
        print("Skipping redshift bin!", flush=True)
        continue

    # Compute sigma crit inverse
    # For photo_z...
    sigma_crit_inv_photo_z = df.photo_z.map_partitions(
        sigma_crit_helper,
        z_L=z_L,
        cosmo=cosmo,
        meta=pd.Series(name="scrit_inv_pz", dtype="float64"),
    )
    # For zmc...
    sigma_crit_inv_zmc = df.zmc.map_partitions(
        sigma_crit_helper,
        z_L=z_L,
        cosmo=cosmo,
        meta=pd.Series(name="scrit_inv_pz", dtype="float64"),
    )

    # Compute and assign weight column
    w1 = df.weight * sigma_crit_inv_photo_z
    w2 = df.weight * df.r * sigma_crit_inv_zmc * sigma_crit_inv_photo_z
    df2 = df.assign(w1=w1, w2=w2)

    # Save to CSV
    # First let's save for w1
    df2.to_csv(
        f"{OUTPUT_DIR}/lensing_w1_bin_{i}.csv",
        single_file=True,
        index=False,
        header=False,
        columns=["ra", "dec", "e_1", "e_2", "w1"],
    )
    # Now let's save for w2
    df2.to_csv(
        f"{OUTPUT_DIR}/lensing_w2_bin_{i}.csv",
        single_file=True,
        index=False,
        header=False,
        columns=["ra", "dec", "e_1", "e_2", "w2"],
    )
    # And finally for w3 (no sigma crit correction)
    df2.to_csv(
        f"{OUTPUT_DIR}/lensing_small_subselection_w3_bin_{i}.csv",
        single_file=True,
        index=False,
        header=False,
        columns=["ra", "dec", "e_1", "e_2", "weight"],
    )
    del (
        df,
        sigma_crit_inv_photo_z,
        sigma_crit_inv_zmc,
        w1,
        w2,
        df2,
    )
    gc.collect()
