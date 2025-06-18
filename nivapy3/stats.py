import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm, theilslopes, pearsonr
from sklearn.preprocessing import StandardScaler


def mk_test(df, col, alpha=0.05):
    """Adapted from http://pydoc.net/Python/ambhas/0.4.0/ambhas.stats/
    by Sat Kumar Tomer.

    Perform the Mann-Kendall test for monotonic trends. Uses the "normal
    approximation" to determine significance and therefore should
    only be used if the number of values is >= 10.

    NOTE: Missing data will be dropped before analysis.

    Args:
        df:    Dataframe. Assumed to be already in sorted order
        col:   Str. Column name in df to use for test
        alpha: Float. Significance level

    Returns:
        Dataframe. Includes the following statistics:

            var_s: Variance of test statistic
            s:     M-K test statistic
            z:     Normalised test statistic
            p:     p-value of the significance test
            trend: Whether to reject the null hypothesis (no trend) at
                   the specified significance level. One of:
                   'increasing', 'decreasing' or 'no trend'
    """
    # Get data
    n_recs = len(df)
    mk_df = df[[col]].dropna(how="any")
    assert len(mk_df) > 1, "Data series must have two or more non-null values."
    if len(mk_df) < n_recs:
        print(
            "WARNING: The data series contains missing values. These will be ignored."
        )
    x = mk_df[col].values

    # Perform test
    n = len(x)
    if n < 10:
        print(
            "WARNING: The data series has fewer than 10 non-null values. "
            "Significance estimates may be unreliable."
        )

    # calculate S
    s = 0
    for k in range(n - 1):
        for j in range(k + 1, n):
            s += np.sign(x[j] - x[k])

    # calculate the unique data
    unique_x = np.unique(x)
    g = len(unique_x)

    # calculate the var(s)
    if n == g:  # there is no tie
        var_s = (n * (n - 1) * (2 * n + 5)) / 18.0
    else:  # there are some ties in data
        tp = np.zeros(unique_x.shape)
        for i in range(len(unique_x)):
            tp[i] = sum(unique_x[i] == x)
        # Sat Kumar's code has "+ np.sum", which is incorrect
        var_s = (
            n * (n - 1) * (2 * n + 5) - np.sum(tp * (tp - 1) * (2 * tp + 5))
        ) / 18.0

    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s == 0:
        z = 0
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = np.nan

    # calculate the p_value
    p = 2 * (1 - norm.cdf(abs(z)))  # two tail test
    h = abs(z) > norm.ppf(1 - alpha / 2.0)

    if (z < 0) and h:
        trend = "decreasing"
    elif (z > 0) and h:
        trend = "increasing"
    elif np.isnan(z):
        trend = np.nan
    else:
        trend = "no trend"

    # Format results
    res_df = pd.DataFrame(
        {
            "description": [
                "Variance of test statistic",
                "M-K test statistic",
                "Normalised test statistic",
                "p-value of the significance test",
                "Type of trend (if present)",
            ],
            "value": [var_s, s, z, p, trend],
        },
        index=["var_s", "s", "z", "p", "trend"],
    )

    return res_df


def seasonal_regional_mk_sen(
    df, time_col="year", value_col="value", block_col="block", alpha=0.05
):
    """Implementation of seasonal/regional Mann-Kendall test based on Helsel and
    Frans (2006; [1]) and the rkt R package [2]. Provides a test for 'overall'
    trends, either for a single site split across multiple seasons, or a
    single region comprising multiple sites.

    The test is the same regardless of whether you're interested in seasonal
    or regional groupings: the 'block-col' argument defines the groups, and
    should contain a unique identifier for either

        (i)  Each season at a single site (fore the seasonal test), or
        (ii) Each site within a single region (for the regional test

    An estimate for the overall Sen's slope (aggregated across all seasons/
    sites) is also provided.

    [1] https://pubs.acs.org/doi/abs/10.1021/es051650b
    [2] https://www.rdocumentation.org/packages/rkt/versions/1.4

    Args:
        df:        Dataframe.
        time_col:  Str. Name of time column. Should contain years or decimal years i.e.
                   Int or Float data type, NOT datetime
        value_col: Str. Name of column containing values of interest
        block_col: Str. Name of column defining blocks (i.e. seasons for the
                   seasonal test or stations for the regional test). Should be Str or
                   Int data type, NOT float
        alpha:     Float. Significance level

    Returns:
        Dataframe. Includes the following statistics:

            var_s: Variance of test statistic
            s:     M-K test statistic
            z:     Normalised test statistic
            p:     p-value of the significance test
            sslp:  Median slope estimate across all seasons/sites
            trend: Whether to reject the null hypothesis (no trend) at
                   the specified significance level. One of:
                   'increasing', 'decreasing' or 'no trend'
    """
    # Check input
    assert df[time_col].dtype in (int, float), (
        "'time_col' must be year or decimal years " "(i.e. Int or Float data type)."
    )

    assert df[block_col].dtype in (int, object), (
        "'block_col' must be categorical " "(i.e. Int or Object data type)."
    )

    df = df.copy()

    # Check blocks
    if pd.isna(df[block_col]).sum() > 0:
        print("WARNING: 'block_col' contains NaN. These rows will be ignored.")
        df = df.dropna(subset=block_col)

    # Containers for output
    res_dict = {
        "block": [],
        "s": [],
        "var_s": [],
    }
    sen_slps = []

    # Loop over blocks
    for block in df[block_col].unique():
        block_df = df.query("%s == @block" % block_col)
        block_df = block_df.set_index(time_col).sort_index()
        block_df = block_df[[value_col]]

        # M-K test
        mk_res = mk_test(block_df, value_col, alpha=alpha)

        # Sen's slopes. Must be calculated manually, as we need the median of all
        # pairwise slopes from all seasons/regions (NOT the median of medians for
        # each season/region)
        block_df = block_df[[value_col]].dropna(how="any")
        xs = block_df.index.values
        vals = block_df[value_col].values
        n = len(block_df)
        for k in range(n - 1):
            for j in range(k + 1, n):
                slp = (vals[j] - vals[k]) / (xs[j] - xs[k])
                sen_slps.append(slp)

        res_dict["block"].append(block)
        res_dict["s"].append(mk_res.loc["s"]["value"])
        res_dict["var_s"].append(mk_res.loc["var_s"]["value"])

    # Aggregate results
    res_df = pd.DataFrame(res_dict)
    s = res_df["s"].sum()
    var_s = res_df["var_s"].sum()
    sslp = np.median(np.array(sen_slps))

    # Convert S to z-scores
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s == 0:
        z = 0
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = np.nan

    # Calculate the p-value
    p = 2 * (1 - norm.cdf(abs(z)))  # two tail test
    h = abs(z) > norm.ppf(1 - alpha / 2.0)

    if (z < 0) and h:
        trend = "decreasing"
    elif (z > 0) and h:
        trend = "increasing"
    elif np.isnan(z):
        trend = np.nan
    else:
        trend = "no trend"

    # Format results
    res_df = pd.DataFrame(
        {
            "description": [
                "M-K test statistic",
                "Normalised test statistic",
                "Variance of test statistic",
                "p-value of the significance test",
                "Overall Sen's slope estimate",
                "Type of trend (if present)",
            ],
            "value": [s, z, var_s, p, sslp, trend],
        },
        index=["s", "z", "var_s", "p", "sslp", "trend"],
    )

    return res_df


def sens_slope(df, value_col, index_col=None, alpha=0.05):
    """Sen's slope test. A non-parametric test for a linear trend. Based on
    scipy's theilslopes function:

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.theilslopes.html#scipy.stats.theilslopes

    Args:
        df:
        value_col: Str. Column name in df with values to test
        index_col: Str or Pandas numerical index. Optional. Column providing x-values
                   for each y-value. If None, values in 'value_col' are assumed to be
                   evenly spaced
        alpha:     Float. Significance level

    Returns:
        Tuple of dataframes (res_df, sen_df). 'sen_df' contains the data used for slope
        estimation. 'res_df' includes the following statistics:

            sslp:  Median slope estimate
            icpt:  Estimated intercept
            lb:    Lower bound on slope estimate at specified alpha
            ub:    Upper bound on slope estimate at specified alpha
            trend: Whether to reject the null hypothesis (no trend) at
                   the specified significance level. One of:
                   'increasing', 'decreasing' or 'no trend'
    """
    # Get data
    n_recs = len(df)
    if isinstance(index_col, pd.core.indexes.base.Index):
        # Index already set
        df2 = df.copy()
    elif isinstance(index_col, str):
        # Set col as index
        df2 = df.set_index(index_col)
    else:
        # Assume evenly spaced
        df2 = df.reset_index()

    sen_df = df2[[value_col]].dropna(how="any")
    assert len(sen_df) > 1, "Data series must have two or more non-null values."
    if len(sen_df) < n_recs:
        print(
            "WARNING: The data series contains missing values. These will be ignored."
        )

    # Perform test
    sslp, icpt, lb, ub = theilslopes(sen_df[value_col].values, sen_df.index, alpha)

    # Get trend
    if (np.sign(lb) == -1) and (np.sign(ub) == -1):
        trend = "decreasing"
    elif (np.sign(lb) == 1) and (np.sign(ub) == 1):
        trend = "increasing"
    else:
        trend = "no trend"

    # Format results
    res_df = pd.DataFrame(
        {
            "description": [
                "Median slope estimate",
                "Estimated intercept",
                "Lower bound on slope estimate at specified alpha",
                "Upper bound on slope estimate at specified alpha",
                "Type of trend (if present)",
            ],
            "value": [sslp, icpt, lb, ub, trend],
        },
        index=["sslp", "icpt", "lb", "ub", "trend"],
    )

    return (res_df, sen_df)


def adjust_lod_values(df, date_col="sample_date", agg_freq="none"):
    """Adjusts LOQ/LOD values according to the OSPAR methodology. Values values are adjusted
    according to the proportion of values in the year (or in the data series as a whole)
    that are at or below the LOQ/LOD:

        xadj = xlod * (1 − p)

    where xlod is the detection limit, p is the proportion of measurements at or below the
    LOD, and xadj is the adjusted value. The effect is that LOD concentrations are reduced
    linearly (from the detection limit to zero) according to how many of the year's (or
    whole series') samples are also at or below the LOD i.e. if all samples are below the
    LOD, estimated concentrations are zero; if almost all samples are above the LOD, any
    LOD values that do occur are assumed to be close to the detection limit itself.

    Args:
        df:       Dataframe of water chemistry data. Must contain numeric parameter columns
                  named 'par_unit' and corresponding columns named 'par_flag' with '<' LOD
                  flags. If 'agg_freq' is not 'none', must also contain a datetime column of
                  sample dates
        date_col: Str. Column in 'df' containing samples dates. Only required if 'agg_freq' is
                  not 'none'
        agg_freq: Str. One of ['none', 'A']. If 'none', the proportion of LOD values for each
                  parameter is calculated for the entire series; if 'A', the proportion is
                  calculated separately for each parameter for each year (i.e. a different
                  correction factor is calculated for each year)

    Returns:
        Dataframe. LOQ/LOD values are adjusted according to the method described above. The
        'par_flag' columns are also removed to avoid confusion (as they no longer apply to the
        values in the dataframe).
    """
    df = df.copy()

    # Get list of chem cols
    cols = df.columns
    par_unit_list = [i for i in cols if i.split("_")[1] != "flag"]
    par_unit_list = [i for i in par_unit_list if i != date_col]

    for par_unit in par_unit_list:
        # Check each 'par_unit' columns also has a 'par_flag' column
        par = par_unit.split("_")[0]
        assert f"{par}_flag" in df.columns, f"Could not find column '{par}_flag'."

        df["isnum"] = ~pd.isna(df[par_unit])
        df["islod"] = df[f"{par}_flag"] == "<"

        if agg_freq == "none":
            # One correction factor for the whole series (for each parameter)
            nvals = df["isnum"].sum()
            nlods = df["islod"].sum()
            plods = nlods / nvals
            df[par_unit] = np.where(
                df[f"{par}_flag"] == "<", df[par_unit] * (1 - plods), df[par_unit]
            )
            df.drop([f"{par}_flag", "isnum", "islod"], inplace=True, axis="columns")

        elif agg_freq == "A":
            # Count number of values and LODs within each year
            df["year"] = df[date_col].dt.year
            cnt_df = df.groupby("year").sum().reset_index()
            cnt_df["plod"] = cnt_df["islod"] / cnt_df["isnum"]
            df = pd.merge(df, cnt_df[["year", "plod"]], how="left", on="year")
            df[par_unit] = np.where(
                df[f"{par}_flag"] == "<", df[par_unit] * (1 - df["plod"]), df[par_unit]
            )
            df.drop(
                [f"{par}_flag", "isnum", "islod", "plod", "year"],
                inplace=True,
                axis="columns",
            )

        else:
            raise ValueError("'agg_freq' must be one of ['A', 'none'].")

    return df


def estimate_fluxes(
    q_df,
    chem_df,
    base_freq="D",
    agg_freq="A",
    method="linear_interpolation",
    st_date=None,
    end_date=None,
    plot_fold=None,
):
    """Takes dataframes of discharge (in m3/s) and water chemistry (in units of mg/l, ug/l and/or ng/l) and estimates
    river fluxes (also known as loads). The API is rather limited at present, and should be improved. Currently
    implements the following methods of flux estimation:

        - linear_interpolation. chem_df is resampled to the same base frequency as q_df and any data gaps are patched
          by linear interpolation. Fluxes are then calculated for each time step at the base frequency and summed to
          the desired aggregation frequency. Simple, but biased unless you have fairly high-frequency,
          regular sampling of water chemistry

        - simple_means. chem_df and q_df are resampled to the specified aggregation frequency by taking means.
          Fluxes for each aggregated timestep are then calculated. Simple, but often biased. Does not make very
          efficient use of the available data

        - log_log_linear_regression. Performs OLS linear regression of log(C) vs log(Q) for dates at the base
          frequency when paired measurments are available. The fitted regression equation is then used to estimate
          C from Q for all times in q_df (at the base frequency). Fluxes are then calculated for each time step and
          summed to the desired aggregation frequency. Prints a warning if R2 from the regression is less than 0.5
          and includes a correction for back-transformation bias in the log-log regression. Works well, assuming a
          reasonable linear relationship exists between log(C) and log(Q)

        - ospar_annual. Calculate annual fluxes according to the OSPAR methodology (see here
          https://nbviewer.jupyter.org/github/JamesSample/rid/blob/master/notebooks/rid_data_exploration.ipynb#2.3.-Calculate-loads
          for details). The OSPAR method is a kind of "ratio estimator" with some tweaks that mean it can ONLY be
          used to calculate annual fluxes.

    Dataset requirements:

        - q_df and chem_df must both have date-time indexes

        - q_df must include a column of flow data named 'flow_m3/s'

        - chem_df must ONLY include columns of concentration data named using the convention 'parname_unit', where
          parname must not include underscores. At present, the only units supported are 'mg/l', 'ug/l' and 'ng/l'

        - chem_df can contain data for multiple parameters (one per column). Fluxes will be estimated for all
          parameters in chem_df

    Args:
        q_df:      Dataframe. Must have a datetime index and a column named 'flow_m3/s'
        chem_df:   Dataframe. Must have a datetime index and each column named 'parname_unit' - see above
        base_freq: Str. 'D' for daily; 'M' for monthly; 'A' for annual etc. The initial data frequency for q_df and
                   chem_df
        agg_freq:  Str. 'D' for daily; 'M' for monthly; 'A' for annual etc. The frequency at which fluxes should be
                   reported
        method:    Str. One of ['linear_interpolation', 'simple_means', 'log_log_linear_regression', 'ospar_annual']
        st_dt:     Str. Only consider values after this date. Format: 'YYYY-MM-DD'
        end_dt:    Str. Only consider values before this date. Format: 'YYYY-MM-DD'
        plot_fold: Raw str. Only relevant if method='log_log_linear_regression'. Folder in which to save diagnostic
                   plots from the log-log regression

    Returns
        Dataframe of fluxes
    """
    # Check inputs
    if plot_fold:
        assert (
            method == "log_log_linear_regression"
        ), "'plot_fold' is only valid for method='log_log_linear_regression'."

    assert isinstance(q_df.index, pd.DatetimeIndex), (
        "'q_df' does not have a datetime index. "
        "Try q_df.index = q_df['my_datetime_column'] first?"
    )

    assert isinstance(chem_df.index, pd.DatetimeIndex), (
        "'chem_df' does not have a datetime index. "
        "Try chem_df.index = chem_df['my_datetime_column'] first?"
    )

    assert (
        "flow_m3/s" in q_df.columns
    ), "'q_df' must include a column named 'flow_m3/s.'"

    for col in chem_df.columns:
        assert col.split("_")[1] in ("mg/l", "ug/l", "µg/l", "ng/l"), (
            f"Could not interpret unit for column '{col}'. "
            "Use syntax 'par_unit' and unit in ('mg/l', 'ug/l', 'µg/l', ng/l')."
        )
    # Sort indexes
    q_df = q_df.copy()["flow_m3/s"].sort_index()
    chem_df = chem_df.copy().sort_index()

    # Check all numeric
    pd.to_numeric(q_df)
    for col in chem_df.columns:
        pd.to_numeric(chem_df[col])

    # Check date ranges
    if st_date:
        assert q_df.index[0] < pd.to_datetime(
            st_date
        ), "'st_date' must be after the first value in 'q_df'."
    if end_date:
        assert q_df.index[-1] > pd.to_datetime(
            end_date
        ), "'end_date' must be before the last value in 'q_df'."

    # Truncate to requested dates
    q_df = q_df.truncate(before=st_date, after=end_date)
    chem_df = chem_df.truncate(before=st_date, after=end_date)

    # Resample to base frequency
    q_df = q_df.resample(base_freq).mean()
    chem_df = chem_df.resample(base_freq).mean()

    # Join
    df = pd.merge(q_df, chem_df, how="left", left_index=True, right_index=True)

    # Calculate fluxes
    if method == "linear_interpolation":
        df = fluxes_linear_interp(df)
        df = df.resample(agg_freq).sum()

        return df

    elif method == "log_log_linear_regression":
        df = fluxes_log_log_linear_regression(df, plot_fold=plot_fold)
        df = df.resample(agg_freq).sum()

        return df

    elif method == "simple_means":
        df = fluxes_simple_means(df, agg_freq=agg_freq)

        return df

    elif method == "ospar_annual":
        assert (
            agg_freq == "A"
        ), "Method 'ospar_annual' can only be used to esimate annual fluxes."

        assert base_freq == "D", "Method 'ospar_annual' requires 'base_freq == 'D'."

        df = fluxes_ospar_annual(df)

        return df

    else:
        raise ValueError("Calculation method not recognised.")


def fluxes_simple_means(df, agg_freq="A"):
    """Called by estimate_fluxes."""
    # Convert flow to volume
    n_secs = (df.index[1] - df.index[0]).total_seconds()
    df["flow_m3"] = df["flow_m3/s"] * n_secs
    del df["flow_m3/s"]

    # Calculate fluxes
    chem_cols = list(df.columns)
    chem_cols.remove("flow_m3")

    # Aggregate. Sum flows, average concs
    agg_dict = {i: "mean" for i in chem_cols}
    agg_dict["flow_m3"] = "sum"
    df = df.resample(agg_freq).agg(agg_dict)

    # Dict of unit conversions
    unit_conv_dict = {
        "mg/l": 1e6,  # mg => kg
        "ug/l": 1e9,  # ug => kg
        "µg/l": 1e9,  # ug => kg
        "ng/l": 1e12,  # ng => kg
    }

    # Loop over chem pars
    for col in chem_cols:
        par, unit = col.split("_")
        df["%s_kg" % par] = df["flow_m3"] * df[col] * 1000 / unit_conv_dict[unit]
        del df[col]

    return df


def fluxes_linear_interp(df):
    """Called by estimate_fluxes."""
    n_nan = df["flow_m3/s"].isna().sum()
    if n_nan > 0:
        print(
            "Warning: 'q_df' has missing values. These will be patched using linear interpolation and "
            "forward-/back-filling, as necessary."
        )

    # Interpolate
    df.interpolate(kind="linear", inplace=True)
    df.fillna(method="backfill", inplace=True)

    # Convert flow to volume
    n_secs = (df.index[1] - df.index[0]).total_seconds()
    df["flow_m3"] = df["flow_m3/s"] * n_secs
    del df["flow_m3/s"]

    # Calculate fluxes
    chem_cols = list(df.columns)
    chem_cols.remove("flow_m3")

    # Dict of unit conversions
    unit_conv_dict = {
        "mg/l": 1e6,  # mg => kg
        "ug/l": 1e9,  # ug => kg
        "µg/l": 1e9,  # ug => kg
        "ng/l": 1e12,  # ng => kg
    }

    # Loop over chem pars
    for col in chem_cols:
        par, unit = col.split("_")
        df["%s_kg" % par] = df["flow_m3"] * df[col] * 1000 / unit_conv_dict[unit]
        del df[col]

    return df


def fluxes_log_log_linear_regression(df, plot_fold=None):
    """Called by estimate_fluxes."""
    df2 = df.copy()

    # Convert zero => NaN
    if np.count_nonzero(df2 == 0) > 0:
        print(
            "Warning: Dataframe contains zeros. These will be converted to NaN before taking logs."
        )
        df[df == 0] = np.nan

    # Patch missing in flow series
    n_nan = df["flow_m3/s"].isna().sum()
    if n_nan > 0:
        print(
            "Warning: 'q_df' has missing values. These will be patched using linear interpolation and "
            "forward-/back-filling, as necessary."
        )

        # Interpolate
        df["flow_m3/s"].interpolate(kind="linear", inplace=True)
        df["flow_m3/s"].fillna(method="backfill", inplace=True)

    # Get logged data (base-10)
    df2 = np.log10(df)

    # Regression
    chem_cols = list(df2.columns)
    chem_cols.remove("flow_m3/s")

    # Containers for results
    res_dict = {
        "param": [],
        "slope": [],
        "intercept": [],
        "r_squared": [],
    }

    for col in chem_cols:
        # OLS regression
        res = smf.ols(formula='Q("%s") ~ Q("flow_m3/s")' % col, data=df2).fit()

        if plot_fold:
            # Plot diagnotsics
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

            axes[0].plot(res.model.exog[:, 1], res.resid, "ro")
            axes[0].set_xlabel("log[Flow (m3/s)]")
            axes[0].set_ylabel("Residual")

            sn.distplot(res.resid.values, ax=axes[1])
            axes[1].set_xlabel("Residual")

            plt.suptitle(col)
            plt.tight_layout()

            # Save png
            png_path = os.path.join(plot_fold, col.split("_")[0] + ".png")
            plt.savefig(png_path, dpi=300)
            plt.close()

        if res.rsquared < 0.5:
            print(
                f"Warning: R2 for regression between log({col}) and log(Q) is < 0.5. "
                "Consider using a different method?"
            )

        # Calc C, inc. correction back-transformation bias. See here:
        # https://nbviewer.jupyter.org/github/JamesSample/martini/blob/master/notebooks/process_norway_chem.ipynb#3.-Concentration-discharge-relationships
        concs = (10 ** res.params[0]) * (df["flow_m3/s"] ** res.params[1])

        # Bias correction
        alpha = np.exp(2.651 * ((res.resid.values) ** 2).mean())
        concs = alpha * concs

        # Update series
        df[col] = df[col].fillna(concs)

        # Add to results
        res_dict["param"].append(col)
        res_dict["slope"].append(res.params[1])
        res_dict["intercept"].append(res.params[0])
        res_dict["r_squared"].append(res.rsquared)

    print("Regression results:")
    res_df = pd.DataFrame(res_dict)
    print(res_df)

    assert np.count_nonzero(np.isnan(df)) == 0

    # Convert flow to volume
    n_secs = (df.index[1] - df.index[0]).total_seconds()
    df["flow_m3"] = df["flow_m3/s"] * n_secs
    del df["flow_m3/s"]

    # Calculate fluxes
    chem_cols = list(df.columns)
    chem_cols.remove("flow_m3")

    # Dict of unit conversions
    unit_conv_dict = {
        "mg/l": 1e6,  # mg => kg
        "ug/l": 1e9,  # ug => kg
        "µg/l": 1e9,  # ug => kg
        "ng/l": 1e12,  # ng => kg
    }

    # Loop over chem pars
    for col in chem_cols:
        par, unit = col.split("_")
        df["%s_kg" % par] = df["flow_m3"] * df[col] * 1000 / unit_conv_dict[unit]
        del df[col]

    return df


def fluxes_ospar_annual(df):
    """Called by estimate_fluxes."""
    # Annual flow volumes
    ann_q_df = (df[["flow_m3/s"]] * 24 * 60 * 60).resample("A").sum()
    ann_q_df.columns = ["flow_m3"]
    ann_q_df.index = ann_q_df.index.year

    # Dict of unit conversions
    unit_conv_dict = {
        "mg/l": 1e6,  # mg => kg
        "ug/l": 1e9,  # ug => kg
        "µg/l": 1e9,  # ug => kg
        "ng/l": 1e12,  # ng => kg
    }

    # Get pars and years to process
    par_unit_list = [i for i in df.columns if i != "flow_m3/s"]
    years = list(df.index.year.unique())

    # Setup container for output
    data_dict = {"year": years}
    for par_unit in par_unit_list:
        par, unit = par_unit.split("_")
        data_dict[f"{par}_kg"] = []

    # Loop over years and pars
    for year in years:
        yr_df = df[df.index.year == year]

        for par_unit in par_unit_list:
            par, unit = par_unit.split("_")

            # Get conc and flow on days where concs were measured
            par_yr_df = yr_df[["flow_m3/s", par_unit]].dropna(subset=[par_unit])

            if len(par_yr_df) == 0:
                # No data for this par and year
                data_dict[f"{par}_kg"].append(np.nan)

            else:
                # Calc intervals t_i
                # Get sample dates
                dates = list(par_yr_df.index.values)

                # Add st and end of year to dates
                dates.insert(0, np.datetime64("%s-01-01" % year))
                dates.append(np.datetime64("%s-01-01" % (year + 1)))
                dates = np.array(dates)

                # Calc differences in seconds between dates and divide by 2
                secs = (np.diff(dates) / np.timedelta64(1, "s")) / 2.0

                # Add "before" and "after" intervals to df
                t_wts = secs[1:] + secs[:-1]

                # The first sample covers the entire period from the start of the
                # year to the sampling date (not from halfway through the period
                # like the rest. The same is true for the last sample to the year
                # end => add half a period to start and end
                t_wts[0] = t_wts[0] + secs[0]
                t_wts[-1] = t_wts[-1] + secs[-1]

                # Add to df
                par_yr_df["t_i"] = t_wts

                # Estimate loads
                # Total flow per time period
                par_yr_df["Qi_ti"] = par_yr_df["flow_m3/s"] * par_yr_df["t_i"]

                # Flux per time period
                par_l = par_unit.split("/")[0]
                par_yr_df[par_l] = (
                    1000
                    * par_yr_df["flow_m3/s"]
                    * par_yr_df["t_i"]
                    * par_yr_df[par_unit]
                )

                # Flow totals
                sum_df = par_yr_df.sum()
                sigma_Qi_Ci_ti = sum_df[par_l]  # Numerator
                sigma_Qi_ti = sum_df["Qi_ti"]  # Denominator
                sigma_q = ann_q_df.loc[year, "flow_m3"]  # Total annual flow

                # Get unit and conv factor
                conv_fac = unit_conv_dict[unit]

                # Estimate flux
                data_dict[f"{par}_kg"].append(
                    (sigma_q * sigma_Qi_Ci_ti) / (conv_fac * sigma_Qi_ti)
                )

    l_df = pd.DataFrame(data_dict)
    l_df.set_index("year", inplace=True)

    return l_df


def best_subsets_ols_regression(df, resp_var, exp_vars, standardise=False):
    """Performs all possible regressions involving exp_vars and returns the one with
    the lowest AIC.

    NOTE: This approach is generally a poor choice, since repeatedly comparing
    many models leads to problems with "multiple comparisons" and essentially
    invalidates any p-values. Use with caution!

    Also note that performance will be poor with many 'exp_vars', because this
    function performs an exhaustive search of all possible combinations (rather
    than just some, as with e.g. stepwise regression).

    Args:
        df:          Dataframe
        resp_var:    Str. Response variable. Column name in 'df'
        exp_vars:    List of str. Explanatory variables. Column names in 'df'
        standardise: Bool. Whether to standardise the 'exp_vars' by subtracting the
                     mean and dividing by the standard deviation

    Returns:
        Tuple (model_result_object, scalar). A residuals plot is also shown. The
        result object is for the "best" model found; the 'scalar' is a scikit-learn
        StandardScaler() object that can be used to transform new data for use with
        the returned model.
    """
    y = df[[resp_var]]
    X = df[exp_vars]

    scaler = StandardScaler()
    if standardise:
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    aics = {}
    for k in range(1, len(exp_vars) + 1):
        for variables in itertools.combinations(exp_vars, k):
            preds = X[list(variables)]
            preds = sm.add_constant(preds)
            res = sm.OLS(y, preds).fit()
            aics[variables] = res.aic

    # Get the combination with lowest AIC
    best_vars = list(min(aics, key=aics.get))

    # Print regression results for these vars
    preds = X[list(best_vars)]
    preds = sm.add_constant(preds)
    res = sm.OLS(y, preds).fit()
    print("Regression results for the model with the lowest AIC:\n")
    print(res.summary())

    # Plot best AIC model and residuals
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

    axes[0].plot(df[resp_var], res.fittedvalues, "ro")
    axes[0].plot(df[resp_var], df[resp_var], "k-")
    axes[0].set_xlabel("Observed", fontsize=16)
    axes[0].set_ylabel("Modelled", fontsize=16)

    sn.histplot(res.resid, ax=axes[1], kde=True)
    axes[1].set_xlabel("Residual", fontsize=16)
    axes[1].set_ylabel("Frequency", fontsize=16)

    plt.tight_layout()

    return (res, scaler)


def double_mad_from_median(data, thresh=3.5):
    """Simple test for outliers in 1D data. Based on the standard MAD approach, but
    modified slightly to allow for skewed datasets. See the example in R here:
    http://eurekastatistics.com/using-the-median-absolute-deviation-to-find-outliers/
    (especially the section "Unsymmetric Distributions and the Double MAD". The
    Python code is based on this post

        https://stackoverflow.com/a/29222992/505698

    See also here

        https://stackoverflow.com/a/22357811/505698

    Args
        data:   Array-like. 1D array of values.
        thresh: Float. Default 3.5. Larger values detect fewer outliers. See the
                section entitled "Z-Scores and Modified Z-Scores" here
                https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm
    Returns
        Array of Bools where ones indicate outliers.
    """
    m = np.nanmedian(data)
    abs_dev = np.abs(data - m)
    left_mad = np.median(abs_dev[data <= m])
    right_mad = np.median(abs_dev[data >= m])
    
    # if (left_mad == 0) or (right_mad == 0):
    #     # Don't identify any outliers. Not strictly correct - see last section of
    #     # https://eurekastatistics.com/using-the-median-absolute-deviation-to-find-outliers/
    #     return np.zeros_like(data, dtype=bool)

    # Replace zero MAD values with a small positive number. Not sure whether this is strictly
    # valid either, but it seems to work quite well and allows flagging of outliers when the
    # left or right MADs are zero
    left_mad = left_mad if left_mad != 0 else 1e-6
    right_mad = right_mad if right_mad != 0 else 1e-6

    data_mad = left_mad * np.ones(len(data))
    data_mad[data > m] = right_mad
    modified_z_score = 0.6745 * abs_dev / data_mad
    modified_z_score[data == m] = 0

    return modified_z_score > thresh


def target_plot(mod, obs, ax=None, title=None):
    """ Target plot comparing normalised bias and normalised, unbiased RMSD between two 
        datasets (usually modelled versus observed). Based on code written by Leah 
        Jackson-Blake for the REFRESH project and described in the REFRESH report as 
        follows:
        
            "The y-axis shows normalised bias between simulated and observed. The 
             x-axis is the unbiased, normalised root mean square difference (RMSD) 
             between simulated and observed data. The distance between a point and the
             origin is total RMSD. RMSD = 1 is shown by the solid circle (any point 
             within this has positively correlated simulated and observed data and 
             positive Nash Sutcliffe scores); the dashed circle marks RMSD = 0.7. 
             Normalised unbiased root mean squared deviation is a useful way of 
             comparing standard deviations of the observed and modelled datasets."
             
        See Joliff et al. (2009) for full details:
        
            https://www.sciencedirect.com/science/article/pii/S0924796308001140
            
    Args:
        mod:   Array-like. 1D array or list of modelled values
        obs:   Array-like. 1D array or list of observed/reference values
        ax:    Matplotlib axis or None. Optional. Axis on which to plot, if desired
        title: Str. Optional. Title for plot
             
    Returns:
        Tuple (normalised_bias, normalised_unbiased_rmsd). Plot is created.
    """
    assert len(mod) == len(obs), "'mod' and 'obs' must be the same length."

    # Convert to dataframe
    df = pd.DataFrame({"mod": np.array(mod), "obs": np.array(obs),})

    # Drop null
    if df.isna().sum().sum() > 0:
        print("Dataset contains some NaN values. These will be ignored.")
        df.dropna(how="any", inplace=True)

    mod = df["mod"].values
    obs = df["obs"].values

    # Calculate stats.
    normed_bias = (mod.mean() - obs.mean()) / obs.std()
    pearson_cc, pearson_p = pearsonr(mod, obs)
    normed_std_dev = mod.std() / obs.std()
    normed_unbiased_rmsd = (
        1.0 + normed_std_dev ** 2 - (2 * normed_std_dev * pearson_cc)
    ) ** 0.5
    normed_unbiased_rmsd = np.copysign(normed_unbiased_rmsd, mod.std() - obs.std())

    # Setup plot
    if ax is None:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, aspect="equal")

    inner_circle = plt.Circle((0, 0), 0.7, edgecolor="k", ls="--", lw=1, fill=False)
    ax.add_artist(inner_circle)

    outer_circle = plt.Circle((0, 0), 1, edgecolor="k", ls="-", lw=1, fill=False)
    ax.add_artist(outer_circle)

    vline = ax.vlines(0, -2, 2)
    hline = ax.hlines(0, -2, 2)

    # Add labels and titles
    ax.set_xlabel("Normalised, unbiased RMSD")
    ax.set_ylabel("Normalised bias")
    if title:
        ax.set_title(title)

    # Plot data
    ax.plot(normed_unbiased_rmsd, normed_bias, "ro", markersize=10, markeredgecolor="k")

    return (normed_bias, normed_unbiased_rmsd)