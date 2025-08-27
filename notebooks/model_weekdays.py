import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    # Weekdays PyMC model: intercept + seasonal effect per weekday
    import polars as pl
    import numpy as np
    import pymc as pm
    import pytensor.tensor as pt
    import matplotlib.pyplot as plt
    import arviz as az
    import marimo as mo

    return az, mo, np, pl, plt, pm, pt


@app.cell
def _(mo):
    mo.md(r"""Zoo Model: Weekday effects""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Read in data

    Add weekday information as Polars enum.
    """
    )
    return


@app.cell(hide_code=True)
def load_data(pl):
    from notebooks.model_baseline import load_data

    # use the data loading from the base model notebook
    _dl_res = load_data.run()[1]
    data = _dl_res["data"]
    closed_dates = _dl_res["closed_dates"]

    # encode weekdays
    weekday_enum = pl.Enum(["ma", "ti", "ke", "to", "pe", "la", "su"])
    data = data.with_columns(pl.col("day_of_week").cast(weekday_enum))
    data_train, data_test = (
        data.filter(
            (pl.col("date") < pl.date(2023, 1, 1))
            & (~pl.col("date").is_in(closed_dates.implode()))
        ),
        data.filter((pl.col("date") >= pl.date(2023, 1, 1))),
    )
    data.head()
    return data_test, data_train, weekday_enum


@app.cell(hide_code=True)
def _(np, plt, weekday_enum):
    from pymc_marketing.prior import Prior
    from pymc_marketing.mmm.fourier import YearlyFourier

    yearly = YearlyFourier(
        n_order=4,
        prior=Prior("Laplace", mu=0, b=0.5, dims=("fourier", "weekdays")),
    )
    prior = yearly.sample_prior(coords={"weekdays": weekday_enum.categories})
    curve = yearly.sample_curve(prior)
    yearly.plot_curve(np.exp(curve), same_axes=True)
    plt.gca()
    return (yearly,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Define the model
    This is the cool stuff!
    """
    )
    return


@app.cell
def _(data_train, pm, pt, weekday_enum, yearly):
    coords = {"date": data_train["date"], "weekdays": weekday_enum.categories}

    with pm.Model(coords=coords) as weekday_model:
        # Define input data
        _day_of_year = pm.Data(
            "day_of_year", data_train["date"].dt.ordinal_day(), dims=("date",)
        )
        _day_of_week = pm.Data(
            "day_of_week", data_train["day_of_week"].to_physical(), dims=("date",)
        )
        _obs_id = pt.arange(_day_of_year.shape[0])

        # Define parameters and their priors
        _avg_daily = pm.Normal("mu", mu=1000, sigma=300, dims=("weekdays"))
        _seasonality = pm.Deterministic(
            "seasonality", yearly.apply(_day_of_year), dims=("date", "weekdays")
        )
        _shape = pm.HalfNormal("shape", sigma=2, dims=("weekdays"))

        # Likelihood
        pm.NegativeBinomial(
            "visitors",
            mu=_avg_daily[_day_of_week] * pt.exp(_seasonality[_obs_id, _day_of_week]),
            alpha=_shape[_day_of_week],
            observed=data_train["visitors"],
            dims=("date",),
        )
    return (weekday_model,)


@app.cell
def _(weekday_model):
    weekday_model
    return


@app.cell(hide_code=True)
def _(pm, weekday_model):
    with weekday_model:
        trace = pm.sample(nuts_sampler="nutpie")
        pm.sample_posterior_predictive(trace, extend_inferencedata=True)
    return (trace,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Look at the results""")
    return


@app.cell
def _(az, plt, trace):
    az.plot_forest(trace, var_names=["mu"])
    plt.gca()
    return


@app.cell
def _(az, plt, trace):
    az.plot_forest(trace, var_names=["shape"])
    plt.gca()
    return


@app.cell(hide_code=True)
def _(data_test, pm, trace, weekday_model):
    with weekday_model:
        pm.set_data(
            {
                "day_of_year": data_test["date"].dt.ordinal_day(),
                "day_of_week": data_test["day_of_week"].to_physical(),
            },
            coords={"date": data_test["date"]},
        )
        predicted = pm.sample_posterior_predictive(trace, predictions=True).predictions
    return (predicted,)


@app.cell
def _(data_test, data_train, predicted, trace):
    from notebooks.model_baseline import plot_posterior_predictions

    plot_posterior_predictions(
        data_train, data_test, trace.posterior_predictive, predicted
    )
    return


@app.cell
def _(az, data_train, np, plt, trace, weekday_enum):
    _fig, _ax = plt.subplots(nrows=7)
    for _idx, _wd in enumerate(weekday_enum.categories.to_list()):
        az.plot_hdi(
            data_train["date"],
            y=trace.posterior["mu"]
            .sel(weekdays=_wd)
            .expand_dims(dim={"date": data_train["date"].to_numpy()}, axis=-1)
            * np.exp(
                trace.posterior["seasonality"]
                .sel(weekdays=_wd)
                .assign_coords({"date": data_train["date"].to_numpy()})
            ),
            smooth=False,
            ax=_ax[_idx],
        )
        _ax[_idx].set_title(_wd)
    plt.show()
    return


@app.cell
def _(data_test, data_train, plt, predicted, trace):
    from notebooks.model_baseline import plot_errors

    plot_errors(
        trace.posterior_predictive["visitors"],
        data_train,
        predicted["visitors"],
        data_test,
    )
    plt.gca()
    return


if __name__ == "__main__":
    app.run()
