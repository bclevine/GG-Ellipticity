import pyarrow.dataset as ds
import numpy as np
from astropy import units as u
import os


def check(files=None, dirs=None):
    """Check that certain files and/or directories exist before executing a script.

    Args:
        files (list of strings, optional): List of paths to files. Defaults to None.
        dirs (list of strings, optional): _description_. Defaults to None.
    """

    if files is not None:
        for file in files:
            assert os.path.isfile(file), f"{file} is not a valid file."
    if dirs is not None:
        for dir in dirs:
            if not os.path.isdir(dir):
                print(f"Creating {dir}.")
                os.mkdir(dir)
    return True


def subselect_parquet(file_to_query, filter_expression, output_dir, output_name):
    """Query a parquet file and save the output to a new parquet file.

    Args:
        file_to_query (str): Path to the parquet file to query
        filter_expression (pyarrow.compute.Expression): Query expression generated with pyarrow.compute
            e.g., filter_exp = (pc.field("ra") < 40) & (pc.field("ra") > 35)
        output_dir (str): Path to the directory we want to save to
        output_name (str): Name of the file we want to save to, e.g. "redmagic_filter_test_{i}.parquet"
    """
    dataset = ds.dataset(
        file_to_query,
        format="parquet",
    )

    dataset_filtered = dataset.filter(filter_expression)

    ds.write_dataset(
        data=dataset_filtered,
        base_dir=output_dir,
        format="parquet",
        basename_template=output_name,
        existing_data_behavior="overwrite_or_ignore",
    )


def compute_median_redshift(file_to_query, filter_expression, column_name="zredmagic"):
    """Query a (small, memory-loadable) parquet file and return the median of some column (default redshift).

    Args:
        file_to_query (str): Path to the parquet file to query
        filter_expression (pyarrow.compute.Expression): Query expression generated with pyarrow.compute
            e.g., filter_exp = (pc.field("ra") < 40) & (pc.field("ra") > 35)
        column_name (str): Name of the column to compute the median on
    """
    dataset = ds.dataset(
        file_to_query,
        format="parquet",
    )

    dataset_filtered = dataset.filter(filter_expression).to_table()
    return np.nanmedian(dataset_filtered[column_name]), len(dataset_filtered)


def d_S_helper(z_S, cosmo):
    return cosmo.angular_diameter_distance(z_S).to(u.pc)


def d_LS_helper(z_S, z_L, cosmo):
    return cosmo.angular_diameter_distance_z1z2(z_L, z_S).to(u.pc)
