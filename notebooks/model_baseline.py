import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    # Initialization code that runs before all other cells
    from datetime import date
    import polars as pl
    import numpy as np
    import pymc as pm
    import xarray as xr
    import arviz as az
    import pytensor.tensor as pt
    import matplotlib.pyplot as plt
    import marimo as mo


@app.cell(hide_code=True)
def _():
    mo.md(r"""# Zoo Model: Baseline""")
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ## Read in data

    Exclude Covid shutdowns from data.
    """
    )
    return


@app.cell
def load_data():
    # Load cleaned data
    data = pl.read_csv("data/clean/visitors.csv").with_columns(
        pl.col("date").str.strptime(pl.Date, "%Y-%m-%d").alias("date")
    )

    # Create closed_dates as a Python set of date objects
    closed_dates = (
        pl.date_range(date(2020, 3, 17), date(2020, 5, 31), "1d", eager=True)
        .append(pl.date_range(date(2020, 11, 30), date(2021, 5, 2), "1d", eager=True))
        .alias("closed_dates")
    )

    data_train, data_test = (
        data.filter(
            (pl.col("date") < pl.date(2023, 1, 1))
            & (~pl.col("date").is_in(closed_dates.implode()))
        ),
        data.filter((pl.col("date") >= pl.date(2023, 1, 1))),
    )
    return data_test, data_train


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ## Yearly seasonality

    Create a multiplicative seasonal effect.
    We borrow a ready object from [PyMC-Marketing](https://www.pymc-marketing.io/en/latest/).
    """
    )
    return


@app.cell
def _():
    from pymc_marketing.prior import Prior
    from pymc_marketing.mmm.fourier import YearlyFourier

    yearly = YearlyFourier(n_order=4, prior=Prior("Laplace", mu=0, b=0.5))
    prior = yearly.sample_prior()
    curve = yearly.sample_curve(prior)
    yearly.plot_curve(np.exp(curve))
    plt.show()
    return (yearly,)


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ## Define the model
    This is the cool stuff!
    """
    )
    return


@app.cell
def _(data_train, yearly):
    # We can give our dimensions meaningful names and values.
    coords = {
        "date": data_train["date"],
    }

    # Model definition happens inside a with statement.
    with pm.Model(coords=coords) as baseline_model:
        # Define input data
        _day_of_year = pm.Data(
            "day_of_year", data_train["date"].dt.ordinal_day(), dims=("date",)
        )

        # Define parameters and their priors
        _avg_daily = pm.Normal("mu", mu=1000, sigma=300)
        _seasonality = pm.Deterministic(
            "seasonality", yearly.apply(_day_of_year), dims=("date",)
        )
        _shape = pm.HalfNormal("shape", sigma=2)

        pm.NegativeBinomial(
            "visitors",
            mu=_avg_daily * pt.exp(_seasonality),
            alpha=_shape,
            observed=data_train["visitors"],
            dims=("date",),
        )
    return (baseline_model,)


@app.cell
def _(baseline_model):
    baseline_model
    return


@app.cell
def _(baseline_model):
    with baseline_model:
        prior_pred = pm.sample_prior_predictive(var_names=["visitors"]).prior_predictive
    return (prior_pred,)


@app.cell
def _(data_train, prior_pred):
    _ax = az.plot_hdi(
        data_train["date"],
        y=prior_pred["visitors"],
        smooth=False,
        hdi_prob=0.5,
        color="#81a1c1",
        fill_kwargs={"alpha": 1},
    )
    az.plot_hdi(
        data_train["date"],
        y=prior_pred["visitors"],
        smooth=False,
        ax=_ax,
        color="#81a1c1",
    )
    _ax.scatter(data_train["date"], data_train["visitors"], s=0.8, c="#bf616a")
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ## Fit the model or "condition on data"
    """
    )
    return


@app.cell
def _(baseline_model):
    with baseline_model:
        trace = pm.sample(nuts_sampler="nutpie")
        trace = pm.sample_posterior_predictive(trace=trace, extend_inferencedata=True)
    posterior_sampled = True
    return posterior_sampled, trace


@app.cell
def _(trace):
    trace
    return


@app.cell
def _(baseline_model, data_test, trace):
    with baseline_model:
        pm.set_data(
            {"day_of_year": data_test["date"].dt.ordinal_day()},
            coords={"date": data_test["date"]},
        )
        predicted = pm.sample_posterior_predictive(trace, predictions=True).predictions
    return (predicted,)


@app.function
def plot_posterior_predictions(data_train, data_test, posterior_predictive, predicted):
    _fig, _ax = plt.subplots()
    _ax = az.plot_hdi(
        data_train["date"],
        y=posterior_predictive["visitors"],
        smooth=False,
        hdi_prob=0.5,
        color="#81a1c1",
        fill_kwargs={"alpha": 1},
    )
    az.plot_hdi(
        data_train["date"],
        y=posterior_predictive["visitors"],
        smooth=False,
        ax=_ax,
        color="#81a1c1",
    )
    az.plot_hdi(
        data_test["date"],
        y=predicted["visitors"],
        smooth=False,
        hdi_prob=0.5,
        ax=_ax,
        color="#88c0d0",
        fill_kwargs={"alpha": 1},
    )
    az.plot_hdi(
        data_test["date"],
        y=predicted["visitors"],
        smooth=False,
        ax=_ax,
        color="#88c0d0",
    )
    _ax.scatter(data_train["date"], data_train["visitors"], s=0.8, c="#bf616a")
    _ax.scatter(data_test["date"], data_test["visitors"], s=0.8, c="#d08770")
    return _fig


@app.cell
def _(data_test, data_train, predicted, trace):
    plot_posterior_predictions(
        data_train, data_test, trace.posterior_predictive, predicted
    )
    return


@app.cell
def _(data_train, posterior_sampled, trace):
    posterior_sampled
    az.plot_hdi(
        data_train["date"],
        y=np.exp(trace.posterior["seasonality"]),
        smooth=False,
        color="#81a1c1",
    )
    return


@app.function
def plot_errors(pred_train, data_train, pred_test, data_test):
    def compute_errors(pred_train, data_train, pred_test, data_test):
        true_train = xr.DataArray(
            data_train["visitors"], coords={"date": data_train["date"].to_list()}
        )
        true_test = xr.DataArray(
            data_test["visitors"], coords={"date": data_test["date"].to_list()}
        )
        error_train = pred_train - true_train
        error_test = pred_test - true_test
        xar = xr.concat(
            [
                xr.merge(
                    [
                        (np.abs(error_train) / true_train).rename("MAPE").mean("date"),
                        np.abs(error_train).rename("Mean absolute error").mean("date"),
                    ]
                ).expand_dims(dim={"in_out": ["in sample"]}),
                xr.merge(
                    [
                        (np.abs(error_test) / true_test).rename("MAPE").mean("date"),
                        np.abs(error_test).rename("Mean absolute error").mean("date"),
                    ]
                ).expand_dims(dim={"in_out": ["out of sample"]}),
            ],
            join="override",
            dim="in_out",
        )
        return az.convert_to_inference_data(xar)

    fig, ax = az.plot_posterior(
        compute_errors(
            pred_train,
            data_train,
            pred_test,
            data_test,
        ),
        grid=(2, 2),
    )
    return fig


@app.cell
def _(data_test, data_train, predicted, trace):
    plot_errors(
        trace.posterior_predictive["visitors"],
        data_train,
        predicted["visitors"],
        data_test,
    )
    return


if __name__ == "__main__":
    app.run()
