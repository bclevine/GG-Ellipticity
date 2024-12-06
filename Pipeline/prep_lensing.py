from query_parquet_helpers import *
import os
import numpy as np
import gc

# import pyarrow.dataset as ds
import pyarrow.compute as pc
import dask.dataframe as dd
from astropy.cosmology import FlatLambdaCDM
from astropy import constants as const
from astropy import units as u
import pandas as pd

# Config
FILE_TO_QUERY = "../Data.nosync/metacal_parquet/metacal_small_catalog.parquet"
REDMAGIC_FILE_TO_QUERY = (
    "../Data.nosync/redmagic_parquet/redmagic_small_processed.parquet"
)
OUTPUT_DIR = "../Data.nosync/metacal_temp_csv"
# TEMPORARY_DIR = "../Data.nosync/PREP_LENSING_TEMP"
REDSHIFT_BUFFER = 0.15

# Prepare constants
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
G = const.G.to(u.pc / u.Msun * (u.km / u.s) * (u.km / u.s))
c = const.c.to(u.km / u.s)
prefactor = 4 * np.pi * G / c**2

# Define the redshift bins
z_min = 0.15
z_max = 0.90
z_edges = np.linspace(z_min, z_max, 31)

# Perform some simple sanity checks
check(files=[FILE_TO_QUERY, REDMAGIC_FILE_TO_QUERY], dirs=[OUTPUT_DIR])

# Main code
# For testing, we can just set the range to 1
for i in range(30):
    print(f"Processing file {i+1}/30.", flush=True)

    # Select all objects with z > z_max + z_buffer
    filter_exp = [("photo_z", ">", z_edges[i + 1] + REDSHIFT_BUFFER)]
    df = dd.read_parquet(FILE_TO_QUERY, filters=filter_exp)

    # Load in the redmagic data
    redmagic_filter_exp = (pc.field("zredmagic") > z_edges[i]) & (
        pc.field("zredmagic") < z_edges[i + 1]
    )
    z_L, len_redmagic = compute_median_redshift(
        REDMAGIC_FILE_TO_QUERY, redmagic_filter_exp
    )
    if len_redmagic < 5:
        print(f"Length of redmagic sample = {len_redmagic}.", flush=True)
        print("Skipping redshift bin!", flush=True)
        continue
    d_L = cosmo.angular_diameter_distance(z_L).to(u.pc)

    # Compute sigma crit inverse
    # For photo_z...
    d_S_photo_z = df.photo_z.map_partitions(
        d_S_helper, cosmo=cosmo, meta=pd.Series(name="d_s", dtype="float64")
    )
    d_LS_photo_z = df.photo_z.map_partitions(
        d_LS_helper, z_L=z_L, cosmo=cosmo, meta=pd.Series(name="d_ls", dtype="float64")
    )
    sigma_crit_inv_photo_z = prefactor * d_L * d_LS_photo_z / d_S_photo_z
    # For zmc...
    d_S_zmc = df.zmc.map_partitions(
        d_S_helper, cosmo=cosmo, meta=pd.Series(name="d_s", dtype="float64")
    )
    d_LS_zmc = df.zmc.map_partitions(
        d_LS_helper, z_L=z_L, cosmo=cosmo, meta=pd.Series(name="d_ls", dtype="float64")
    )
    sigma_crit_inv_zmc = prefactor * d_L * d_LS_zmc / d_S_zmc

    # Compute and assign weight column
    w1 = df.weight * sigma_crit_inv_photo_z
    w2 = (df.weight * df.r * sigma_crit_inv_zmc * sigma_crit_inv_photo_z) * (
        sigma_crit_inv_zmc
        >= 0
        # df.zmc > z_edges[i + 1] + REDSHIFT_BUFFER
    )  # The last part sets w2=0 if zmc < z_min
    df2 = df.assign(w1=w1, w2=w2)

    # Save to CSV
    # First let's save for w1
    df2.to_csv(
        f"{OUTPUT_DIR}/lensing_small_subselection_w1_bin_{i}.csv",
        single_file=True,
        index=False,
        header=False,
        columns=["ra", "dec", "e_1", "e_2", "w1"],
        # The header is not included because I couldn't figure out how to make it work with treecorr.
        # Instead we will need to keep track of the column indicies and feed those into treecorr.
        # We also need to check whether or not this will overwrite a pre-existing file.
    )
    # Now let's save for w2
    df2.to_csv(
        f"{OUTPUT_DIR}/lensing_small_subselection_w2_bin_{i}.csv",
        single_file=True,
        index=False,
        header=False,
        columns=["ra", "dec", "e_1", "e_2", "w2"],
        # The header is not included because I couldn't figure out how to make it work with treecorr.
        # Instead we will need to keep track of the column indicies and feed those into treecorr.
        # We also need to check whether or not this will overwrite a pre-existing file.
    )
    if i == 29:
        df2.to_csv(
            f"{OUTPUT_DIR}/lensing_small_subselection_bin_HEADER_TEMPLATE.csv",
            single_file=True,
            index=False,
            header=True,
            columns=["ra", "dec", "e_1", "e_2", "w1"],
        )

    del (
        df,
        d_S_photo_z,
        d_LS_photo_z,
        d_S_zmc,
        d_LS_zmc,
        sigma_crit_inv_photo_z,
        sigma_crit_inv_zmc,
        w1,
        w2,
        df2,
    )
    gc.collect()

    # =============================================================
    # OLD CODE
    # =============================================================

    # # Save to temporary parquet file
    # dd.to_parquet(df2, TEMPORARY_DIR, write_index=False)

    # # Recombine the temp parquet files into a single big parquet file
    # data = ds.dataset(TEMPORARY_DIR, format="parquet")
    # ds.write_dataset(
    #     data.replace_schema(data.schema.remove_metadata()),
    #     OUTPUT_DIR,
    #     format="parquet",
    #     basename_template=(f"lensing_small_subselection_bin_{i}" + "_{i}.parquet"),
    #     existing_data_behavior="overwrite_or_ignore",
    # )
    # os.rename(
    #     OUTPUT_DIR + "/" + f"lensing_small_subselection_bin_{i}_0.parquet",
    #     OUTPUT_DIR + "/" + f"lensing_small_subselection_bin_{i}.parquet",
    # )

    # # Reset the temp folder
    # files_to_delete = os.listdir(TEMPORARY_DIR)
    # for f in files_to_delete:
    #     os.remove(TEMPORARY_DIR + "/" + f)


# for i in range(30):
#     print(f"Saving file {i+1}/30")

#     # Select all objects with z > z_max + z_buffer
#     filter = pc.field("z") > (z_edges[i + 1] + REDSHIFT_BUFFER)
#     output = f"lensing_small_subselection_bin_{i}.parquet"

#     # Cycle through all the redshift bins and save a parquet file for each.
#     subselect_parquet(FILE_TO_QUERY, filter, OUTPUT_DIR, output + "_{i}")
