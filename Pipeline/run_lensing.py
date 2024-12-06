import numpy as np
import pyarrow.compute as pc
import treecorr as tc
from run_lensing_helpers import *
from query_parquet_helpers import check
from astropy.cosmology import FlatLambdaCDM
import pickle
import gc

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

# CONFIG
PATCH_FILE = "small_patch_centers.txt"
CSV_OR_PARQUET = "CSV"
LENS_CATALOG = "../Data.nosync/redmagic_parquet/redmagic_small_processed.parquet"
SOURCE_CATALOG_W1 = (
    "../Data.nosync/metacal_temp/lensing_small_subselection_w1_bin_"  # {i}.suffix
)
SOURCE_CATALOG_W2 = (
    "../Data.nosync/metacal_temp/lensing_small_subselection_w2_bin_"  # {i}.suffix
)
RANDOM_CATALOG = "randoms_full_catalog.parquet"
TMP_DIR = "patch_dir_tmp"
OUT_DIR = "/gpfs/scratch/bclevine/out"  # + "/"

# Perform some simple sanity checks
check(files=[PATCH_FILE, LENS_CATALOG, RANDOM_CATALOG], dirs=[TMP_DIR, OUT_DIR])

z_min = 0.15
z_max = 0.90
z_edges = np.linspace(z_min, z_max, 31)

if CSV_OR_PARQUET == "CSV":
    low_mem = True
elif CSV_OR_PARQUET == "PARQUET":
    low_mem = False

all_out = {
    "r": [],
    "numerator": [],
    "numerator_unc": [],
    "numerator_im": [],
    "denominator": [],
    "n_lens": [],
    "cov": [],
    "boost_factor_cov": [],
    # "rg_denom_cov": [],
}
round_out = {
    "r": [],
    "numerator": [],
    "numerator_unc": [],
    "numerator_im": [],
    "denominator": [],
    "n_lens": [],
    "cov": [],
    "boost_factor_cov": [],
    # "rg_denom_cov": [],
}
elliptical_out = {
    "r": [],
    "numerator": [],
    "numerator_unc": [],
    "numerator_im": [],
    "denominator": [],
    "n_lens": [],
    "cov": [],
    "boost_factor_cov": [],
    # "rg_denom_cov": [],
}
randoms_out = {
    "r": [],
    "numerator": [],
    "numerator_unc": [],
    "numerator_im": [],
    "denominator": [],
    "n_lens": [],
    "cov": [],
}
uncorrected_out = {"all": [], "round": [], "elliptical": []}


for i in range(30):
    print(f"Running bin {i+1}/30.", flush=True)

    # Load in the source catalog.
    if CSV_OR_PARQUET == "CSV":
        source_cat_1 = tc.Catalog(
            f"{SOURCE_CATALOG_W1}{i}.csv",
            ra_col=8,
            dec_col=1,
            ra_units="deg",
            dec_units="deg",
            g1_col=2,
            g2_col=3,
            w_col=12,
            patch_centers=PATCH_FILE,
            file_type="ASCII",
            delimiter=",",
            save_patch_dir=TMP_DIR,
        )
        source_cat_2 = tc.Catalog(
            f"{SOURCE_CATALOG_W2}{i}.csv",
            ra_col=8,
            dec_col=1,
            ra_units="deg",
            dec_units="deg",
            g1_col=2,
            g2_col=3,
            w_col=13,
            patch_centers=PATCH_FILE,
            file_type="ASCII",
            delimiter=",",
            save_patch_dir=TMP_DIR,
        )
    elif CSV_OR_PARQUET == "PARQUET":
        source_cat_1 = tc.Catalog(
            f"{SOURCE_CATALOG_W1}{i}.parquet",
            ra_col="ra",
            dec_col="dec",
            ra_units="deg",
            dec_units="deg",
            g1_col="e_1",
            g2_col="e_2",
            w_col="w1",
            patch_centers=PATCH_FILE,
        )
        source_cat_2 = tc.Catalog(
            f"{SOURCE_CATALOG_W2}{i}.parquet",
            ra_col="ra",
            dec_col="dec",
            ra_units="deg",
            dec_units="deg",
            g1_col="e_1",
            g2_col="e_2",
            w_col="w2",
            patch_centers=PATCH_FILE,
        )
    else:
        raise Exception(
            """Source catalog format not recognized (must be either 'CSV' or 'PARQUET')."""
        )

    # Prep for Treecorr.
    min_sep, max_sep = bin_info((z_edges[i] + z_edges[i + 1]) / 2, cosmo)
    config = {
        "sep_units": "degrees",
        "nbins": 15,
        "min_sep": min_sep,
        "max_sep": max_sep,
        "var_method": "jackknife",
        # "bin_slop": 0,
    }

    # Load in the lens catalog.
    redmagic_filter_exp = (pc.field("zredmagic") > z_edges[i]) & (
        pc.field("zredmagic") < z_edges[i + 1]
    )
    round_redmagic_filter_exp = (
        (pc.field("zredmagic") > z_edges[i])
        & (pc.field("zredmagic") < z_edges[i + 1])
        & (pc.field("ellip_bin") == -1)
    )
    elliptical_redmagic_filter_exp = (
        (pc.field("zredmagic") > z_edges[i])
        & (pc.field("zredmagic") < z_edges[i + 1])
        & (pc.field("ellip_bin") == 1)
    )

    # Load in the random catalog.
    randoms_filter_exp = (pc.field("z") > z_edges[i]) & (pc.field("z") < z_edges[i + 1])

    # We need to clear the cached patches.
    source_cat_1.write_patches()
    source_cat_2.write_patches()

    # Run Treecorr and save outputs for each ellipticity bin.
    # Start with the randoms.
    rand_cat, rand_length = create_lens_catalog(
        randoms_filter_exp, RANDOM_CATALOG, PATCH_FILE
    )
    rg1, rg2 = run_treecorr(
        config,
        rand_cat,
        source_cat_1,
        source_cat_2,
        low_mem,
    )
    randoms = extract_outputs(rg1, rg2)
    save_outputs(randoms, randoms_out, rand_length)
    print("Finished randoms.", flush=True)

    # Now move on to the real data.
    # ALL
    lens_cat, all_length = create_lens_catalog(
        redmagic_filter_exp, LENS_CATALOG, PATCH_FILE
    )
    ng1, ng2 = run_treecorr(
        config,
        lens_cat,
        source_cat_1,
        source_cat_2,
        low_mem,
    )
    corrected, uncorrected = extract_outputs(ng1, ng2, rg1, rg2)
    save_outputs(corrected, all_out, all_length, uncorrected, uncorrected_out["all"])
    # Clear memory.
    del lens_cat, ng1, ng2
    gc.collect()
    print("Finished all sample.", flush=True)

    # ROUND
    lens_cat, round_length = create_lens_catalog(
        round_redmagic_filter_exp, LENS_CATALOG, PATCH_FILE
    )
    ng1, ng2 = run_treecorr(
        config,
        lens_cat,
        source_cat_1,
        source_cat_2,
        low_mem,
    )
    corrected, uncorrected = extract_outputs(ng1, ng2, rg1, rg2)
    save_outputs(
        corrected, round_out, round_length, uncorrected, uncorrected_out["round"]
    )
    # Clear memory.
    del lens_cat, ng1, ng2
    gc.collect()
    print("Finished round sample.", flush=True)

    # ELLIPTICAL
    lens_cat, elliptical_length = create_lens_catalog(
        elliptical_redmagic_filter_exp, LENS_CATALOG, PATCH_FILE
    )
    ng1, ng2 = run_treecorr(
        config,
        lens_cat,
        source_cat_1,
        source_cat_2,
        low_mem,
    )
    corrected, uncorrected = extract_outputs(ng1, ng2, rg1, rg2)
    save_outputs(
        corrected,
        elliptical_out,
        elliptical_length,
        uncorrected,
        uncorrected_out["elliptical"],
    )
    # Clear memory.
    del lens_cat, ng1, ng2
    gc.collect()
    print("Finished elliptical sample.", flush=True)

    # Write to file.
    with open(f"{OUT_DIR}/lensing_results_all_{i}.pkl", "wb") as f:
        pickle.dump(all_out, f)
    with open(f"{OUT_DIR}/lensing_results_round_{i}.pkl", "wb") as f:
        pickle.dump(round_out, f)
    with open(f"{OUT_DIR}/lensing_results_elliptical_{i}.pkl", "wb") as f:
        pickle.dump(elliptical_out, f)
    with open(f"{OUT_DIR}/lensing_results_uncorrected_{i}.pkl", "wb") as f:
        pickle.dump(uncorrected_out, f)
    with open(f"{OUT_DIR}/lensing_results_randoms_{i}.pkl", "wb") as f:
        pickle.dump(randoms_out, f)

# Write final results to file.
with open(f"{OUT_DIR}/lensing_results_all.pkl", "wb") as f:
    pickle.dump(all_out, f)
with open(f"{OUT_DIR}/lensing_results_round.pkl", "wb") as f:
    pickle.dump(round_out, f)
with open(f"{OUT_DIR}/lensing_results_elliptical.pkl", "wb") as f:
    pickle.dump(elliptical_out, f)
with open(f"{OUT_DIR}/lensing_results_uncorrected.pkl", "wb") as f:
    pickle.dump(uncorrected_out, f)
with open(f"{OUT_DIR}/lensing_results_randoms.pkl", "wb") as f:
    pickle.dump(randoms_out, f)
