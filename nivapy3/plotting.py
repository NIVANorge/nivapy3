import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st


def plot_water_chemistry(wc_df, titles="station_id", depth1=0, depth2=0):
    """Plot water chemistry data returned by either nivapy.da.select_ndb_water_chemistry()
    or nivapy.da.select_resa_water_chemistry. Creates an interactive 'grid plot', where
    each station occupies a row and each parameter occupies a column.

    Note: Plotting large numbers of stations and/or parameters may take a long time and
    create a messy plot. Consider filtering the dataframe first if this is the case.

    Args:
        wc_df:  Dataframe. Water chemistry dataframe in format returned by
                select_ndb_water_chemistry() or select_resa_water_chemistry()
        titles: Str. Column in wc_df to use for plot titles. By default, plots are labelled
                with the 'station_id'. TAKE CARE if using any other column with data exported
                from the NIVADATABASE - due to the database structure, only 'station_id' is
                guaranteed to be unique.
        depth1: Float. Default 0. Minimum deoth of sample
        depth2: Float. Default 0. Maximum depth of sample

    Returns:
        None. Matplotlib figure is displayed.
    """
    # Get data of interest
    df = wc_df.copy()
    df = df.query("(depth1 == @depth1) and " "(depth2 == @depth2)")

    # Check have data
    if len(df) == 0:
        raise ValueError(
            "The water chemistry dataframe contains no data "
            "for the specified depths (depth1=%s and depth2=%s).\n"
            "Consider specifying a different depth range." % (depth1, depth2)
        )

    cols_to_del = set(
        ["station_id", "station_code", "station_name", "depth1", "depth2"]
    ) - set(
        [
            titles,
        ]
    )
    for col in cols_to_del:
        del df[col]

    # Get pars and units
    par_list = list(set(df.columns) - set([titles, "sample_date"]))

    # Get number of rows (=n_stns) and cols (=n_pars)
    rows = len(df[titles].unique())
    cols = len(par_list)

    if rows > 10:
        print(
            "WARNING: You are attempting to plot data for %s stations. "
            "This could get messy!" % rows
        )

    if cols > 10:
        print(
            "WARNING: You are attempting to plot data for %s parameters. "
            "This could get messy!" % cols
        )

    # Setup plot grid
    fig, axes = plt.subplots(
        nrows=rows,
        ncols=cols,
        sharex=True,
        sharey=False,
        figsize=(12, rows * 2),
        squeeze=False,
    )

    # Lists for min and max dates
    st_dts = []
    end_dts = []

    # Loop over series
    for pid, par_unit in enumerate(par_list):
        for sid, stn in enumerate(list(df[titles].unique())):
            # Extract series
            ts = df.query("%s==@stn" % titles)
            ts = ts[["sample_date", par_unit]]
            ts.dropna(how="any", inplace=True)
            ts.index = ts["sample_date"]
            del ts["sample_date"]
            ts.sort_index(inplace=True)

            # Add start and end dates to lists
            if len(ts) > 0:
                st_dts.append(ts.index[0])
                end_dts.append(ts.index[-1])

                # Plot
                ts.plot(ax=axes[sid, pid], legend=False, marker="o")

            # Get par and unit
            par = "_".join(par_unit.split("_")[:-1])
            unit = par_unit.split("_")[-1]

            # Labels
            axes[sid, pid].set_title("%s at %s" % (par, stn), fontsize=12)
            axes[sid, pid].set_xlabel("")
            if unit:
                axes[sid, pid].set_ylabel("%s (%s)" % (par, unit), fontsize=10)
            else:
                axes[sid, pid].set_ylabel(par, fontsize=14)

    # Loop over series to set axis limits
    min_dt = min(st_dts)
    max_dt = max(end_dts)
    for pid, par_unit in enumerate(par_list):
        for sid, stn in enumerate(list(df[titles].unique())):
            # Set x-limits
            axes[sid, pid].set_xlim([min_dt, max_dt])

    # Tidy
    plt.tight_layout()


def plot_sens_slope(res_df, sen_df, xlabel=None, ylabel=None, title=None):
    """Plot results returned from nivapy.stats.sens_slope().

    Args:
        res_df: Dataframe. Returned by nivapy.stats.sens_slope()
        sen_df: Dataframe. Returned by nivapy.stats.sens_slope()
        xlabel: Str. Label for x-axis
        ylabel: Str. Label for y-axis
        title:  Str. Label for plot title

    Returns:
        None. Figure is displayed in Jupyter Lab.
    """
    # Extract data
    sslp = res_df.loc["sslp"]["value"]
    icpt = res_df.loc["icpt"]["value"]
    lb = res_df.loc["lb"]["value"]
    ub = res_df.loc["ub"]["value"]
    col = sen_df.columns[0]

    # Plot
    fig = plt.figure(figsize=(6, 4))
    plt.plot(sen_df.index.values, sen_df[col].values, "bo-")
    plt.plot(sen_df.index.values, sen_df.index.values * sslp + icpt, "k-")

    # Labels
    if xlabel:
        plt.xlabel(xlabel, fontsize=16)
    if ylabel:
        plt.ylabel(ylabel, fontsize=16)
    if title:
        plt.title(title, fontsize=18)

    plt.tight_layout()


def target_plot(
    mod_list, obs_list, labels=None, markers=None, colours=None, ax=None, title=None
):
    """Target plot comparing normalised bias and normalised, unbiased RMSD between two
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

    Each element in the input lists will create one point on the plots. For
    example:

        target_plot([mod1, mod2, mod3],
                    [obs1, obs2, obs3],
                    labels=['stn1', 'stn2', 'stn3'],
                    markers=['o', '+', '^'],
                    colours=['r', 'g', 'b'],
                   )

    will create a plot with three points - one per station - by comparing mod1 with
    obs1, mod2 with obs2 etc.

    Note that matching is performned based on index position, so check that mod_list
    and obs_list contain elements in the SAME ORDER.

    Args:
        mod_list: List of 1D arrays. Each element of the list contains an array of
                  modelled values
        obs_list: List of 1D arrays. Each element of the list contains an array of
                  modelled values
        labels:   List of strings. Labels for legend
        markers:  List of strings. Matplotlib marker styles
        colours:  List of strings. Matplotlib colour codes
        ax:       Matplotlib axis or None. Optional. Axis on which to plot, if desired
        title:    Str. Optional. Title for plot

    Returns:
        Tuple (normalised_bias, normalised_unbiased_rmsd). Plot is created.
    """
    assert len(mod_list) == len(
        obs_list
    ), "'mod_list' and 'obs_list' must be the same length."

    if labels:
        assert len(mod_list) == len(
            labels
        ), "'labels' must be the same length as 'obs_list'."

    if markers:
        assert len(mod_list) == len(
            labels
        ), "'markers' must be the same length as 'obs_list'."

    if colours:
        assert len(mod_list) == len(
            labels
        ), "'colours' must be the same length as 'obs_list'."

    # Setup plot
    if ax is None:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, aspect="equal")

    inner_circle = plt.Circle((0, 0), 0.7, edgecolor="k", ls="--", lw=1, fill=False)
    ax.add_artist(inner_circle)

    outer_circle = plt.Circle((0, 0), 1, edgecolor="k", ls="-", lw=1, fill=False)
    ax.add_artist(outer_circle)

    # Add labels and titles
    ax.set_xlabel("Normalised, unbiased RMSD")
    ax.set_ylabel("Normalised bias")
    if title:
        ax.set_title(title)

    # Loop over data to compare
    for idx in range(len(obs_list)):
        # Convert to dataframe
        df = pd.DataFrame(
            {
                "mod": np.array(mod_list[idx]),
                "obs": np.array(obs_list[idx]),
            }
        )

        # Drop null
        if df.isna().sum().sum() > 0:
            print(f"Dataset at index {idx} contains NaN values. These will be ignored.")
            df.dropna(how="any", inplace=True)

        mod = df["mod"].values
        obs = df["obs"].values

        # Calculate stats.
        normed_bias = (mod.mean() - obs.mean()) / obs.std()
        pearson_cc, pearson_p = st.pearsonr(mod, obs)
        normed_std_dev = mod.std() / obs.std()
        normed_unbiased_rmsd = (
            1.0 + normed_std_dev**2 - (2 * normed_std_dev * pearson_cc)
        ) ** 0.5
        normed_unbiased_rmsd = np.copysign(normed_unbiased_rmsd, mod.std() - obs.std())

        # Plot data
        if markers:
            marker = markers[idx]
        else:
            marker = "o"

        if colours:
            colour = colours[idx]
        else:
            colour = "r"

        if labels:
            label = labels[idx]
        else:
            label = None

        ax.plot(
            normed_unbiased_rmsd,
            normed_bias,
            marker=marker,
            markersize=10,
            markerfacecolor=colour,
            markeredgecolor="k",
            label=label,
        )

        if labels:
            ax.legend(loc="best")

    return (normed_bias, normed_unbiased_rmsd)