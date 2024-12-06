import numpy as np
import pyarrow.dataset as ds
import pyarrow.compute as pc
import treecorr as tc


def bin_info(z, cosmo):
    """
    For given redshift, find the separation in angular units
    """
    d = (cosmo.comoving_distance(z).value * 10**6) * 0.7 / (1.0 + z)
    min_sep = np.arctan(0.10 * 10**6 / d) * (180.0 / np.pi)
    max_sep = np.arctan(30.0 * 10**6 / d) * (180.0 / np.pi)
    return min_sep, max_sep


def create_lens_catalog(redmagic_filter_exp, LENS_CATALOG, PATCH_FILE, weight=True):
    # Load in the lens catalog.
    redmagic_dataset = ds.dataset(
        LENS_CATALOG,
        format="parquet",
    )
    redmagic_filtered = redmagic_dataset.filter(redmagic_filter_exp)
    length = np.sum(redmagic_filtered.to_table()["weight"])
    # length = len(redmagic_filtered.to_table())
    print("Number of lenses:", length)
    if weight == True:
        return (
            tc.Catalog(
                ra=redmagic_filtered.to_table().combine_chunks()["ra"],
                dec=redmagic_filtered.to_table().combine_chunks()["dec"],
                w=redmagic_filtered.to_table().combine_chunks()["weight"],
                ra_units="deg",
                dec_units="deg",
                patch_centers=PATCH_FILE,
            ),
            length,
        )
    else:
        return (
            tc.Catalog(
                ra=redmagic_filtered.to_table().combine_chunks()["ra"],
                dec=redmagic_filtered.to_table().combine_chunks()["dec"],
                ra_units="deg",
                dec_units="deg",
                patch_centers=PATCH_FILE,
            ),
            length,
        )


def run_treecorr(
    config,
    lens_cat,
    source_cat_1,
    source_cat_2,
    low_mem,
):
    # Run Treecorr.
    ng1 = tc.NGCorrelation(config)
    ng2 = tc.NGCorrelation(config)
    ng1.process(lens_cat, source_cat_1, low_mem=low_mem)
    ng2.process(lens_cat, source_cat_2, low_mem=low_mem)

    return ng1, ng2


def extract_outputs(ng1, ng2, rg1=None, rg2=None):
    """
    Extract the outputs of a set of processed tc correlations.
    """
    r_arr = np.exp(ng1.meanlogr)
    numerator_arr = ng1.xi * ng1.weight
    numerator_unc_arr = np.sqrt(ng1.varxi) * ng1.weight
    numerator_im_arr = ng1.xi_im * ng1.weight
    denominator_arr = ng2.weight

    cov_func = lambda corrs: (corrs[0].xi * corrs[0].weight) / corrs[1].weight
    cov = tc.estimate_multi_cov([ng1, ng2], "jackknife", func=cov_func)

    uncorrected = (
        r_arr,
        numerator_arr,
        numerator_unc_arr,
        numerator_im_arr,
        denominator_arr,
        cov,
    )

    if rg1 is not None:
        ng1.calculateXi(rg=rg1)
        ng2.calculateXi(rg=rg2)
        r_arr = np.exp(ng1.meanlogr)
        numerator_arr = ng1.xi * ng1.weight
        numerator_unc_arr = np.sqrt(ng1.varxi) * ng1.weight
        numerator_im_arr = ng1.xi_im * ng1.weight
        denominator_arr = ng2.weight

        cov_func = lambda corrs: (corrs[0].xi * corrs[0].weight) / corrs[1].weight
        cov = tc.estimate_multi_cov([ng1, ng2], "jackknife", func=cov_func)

        boost_factor_cov_func = lambda corrs: (corrs[0].weight / corrs[1].weight)
        boost_factor_cov = tc.estimate_multi_cov(
            [ng2, rg2], "jackknife", func=boost_factor_cov_func
        )

        # denom_cov_func = lambda ng: ng.weight
        # ng_denom_cov = ng2.estimate_cov("jackknife", func=denom_cov_func)
        # rg_denom_cov = rg2.estimate_cov("jackknife", func=denom_cov_func)

        corrected = (
            r_arr,
            numerator_arr,
            numerator_unc_arr,
            numerator_im_arr,
            denominator_arr,
            cov,
            boost_factor_cov,
        )

        return corrected, uncorrected

    else:
        return uncorrected


def save_outputs(
    corrected,
    corrected_dict,
    corrected_length,
    uncorrected=None,
    uncorrected_dict_arr=None,
):
    if uncorrected is not None:
        (
            r,
            numerator,
            numerator_unc,
            numerator_im,
            denominator,
            cov,
            boost_factor_cov,
        ) = corrected
    else:
        (
            r,
            numerator,
            numerator_unc,
            numerator_im,
            denominator,
            cov,
        ) = corrected
    corrected_dict["r"].append(r)
    corrected_dict["numerator"].append(numerator)
    corrected_dict["numerator_unc"].append(numerator_unc)
    corrected_dict["numerator_im"].append(numerator_im)
    corrected_dict["denominator"].append(denominator)
    corrected_dict["n_lens"].append(corrected_length)
    corrected_dict["cov"].append(cov)
    if uncorrected is not None:
        corrected_dict["boost_factor_cov"].append(boost_factor_cov)
        uncorrected_dict_arr.append(uncorrected)
    return None


def extract_and_save_boost_factor(ng, rg, ng_length, rg_length, output_dict):
    boost_factor_cov_func = lambda corrs: (corrs[0].weight / corrs[1].weight)
    boost_factor_cov = tc.estimate_multi_cov(
        [ng, rg], "jackknife", func=boost_factor_cov_func
    )
    output_dict["r"].append(ng.meanlogr)
    output_dict["ng_weight"].append(ng.weight)
    output_dict["rg_weight"].append(rg.weight)
    output_dict["cov"].append(boost_factor_cov)
    output_dict["ng_length"].append(ng_length)
    output_dict["rg_length"].append(rg_length)
    return None


def sum_normalized_weights(jk_bin, ng, nn_cat):
    jk_mask = [(jk_bin not in key) for key in ng.results.keys()]
    indicies = np.array(list(ng.results.keys()))[jk_mask]
    weights = np.sum(
        np.array([ng.results[tuple(idx)].weight for idx in indicies]), axis=0
    )

    cat_patches = np.array(nn_cat.patches)[np.arange(150) != jk_bin]
    n_obj = np.sum([np.sum(patch.w) for patch in cat_patches])

    return weights / n_obj


def compute_jackknife_bf(ng, nn_cat, rg, rr_cat):
    full_bf = (ng.weight / np.sum(nn_cat.w)) / (rg.weight / rr_cat.nobj)
    ndim = len(full_bf)
    npatch = ng.npatch1
    numerators = np.array(
        [sum_normalized_weights(i, ng, nn_cat) for i in range(npatch)]
    )
    denominators = np.array(
        [sum_normalized_weights(i, ng, nn_cat) for i in range(npatch)]
    )
    bf_jk = numerators / denominators
    diffs = bf_jk - full_bf

    matlist = []
    for i in range(npatch):
        matlist.append(np.broadcast_to(diffs[i], (ndim, ndim)).T * (diffs[i]))
    rhs = np.sum(matlist, axis=0)
    norm = (npatch - 1) / npatch

    return norm * rhs, full_bf


def extract_and_save_DES(ng, rg, lens_cat, rand_cat, output_dict):
    output_dict["r"].append(ng.meanr)
    output_dict["xi"].append(ng.xi)
    output_dict["cov"].append(ng.cov)
    # output_dict["ng_weight"].append(ng.weight)
    # output_dict["rg_weight"].append(rg.weight)

    bfcov, bf = compute_jackknife_bf(ng, lens_cat, rg, rand_cat)
    output_dict["bf"].append(bf)
    output_dict["cov_bf"].append(bfcov)
    # output_dict["ng_length"].append(ng_length)
    # output_dict["rg_length"].append(rg_length)
    return None
