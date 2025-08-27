import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Zoo model: Hourly weather data""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Data imports""")
    return


@app.cell(hide_code=True)
def _():
    # Weather Hourly PyMC model: intercept + seasonal effect per weekday + hourly weather effects with saturation
    import polars as pl
    import numpy as np
    import pymc as pm
    import pytensor.tensor as pt
    import matplotlib.pyplot as plt
    import arviz as az
    import marimo as mo

    return az, mo, np, pl, plt, pm, pt


@app.cell(hide_code=True)
def load_data(pl):
    from notebooks.model_baseline import load_data as load_baseline_data

    # Use the data loading from the base model notebook for visitor data
    _dl_res = load_baseline_data.run()[1]
    data = _dl_res["data"]
    closed_dates = _dl_res["closed_dates"]

    # Encode weekdays
    weekday_enum = pl.Enum(["ma", "ti", "ke", "to", "pe", "la", "su"])
    data = data.with_columns(pl.col("day_of_week").cast(weekday_enum))

    # Load hourly weather data
    weather_hourly = pl.read_csv(
        "data/clean/helsinki_kaisaniemi_hourly.csv"
    ).with_columns(date=pl.col("date").str.to_date("%Y-%m-%d"))

    weather_hourly = weather_hourly.with_columns(
        [
            pl.when(pl.col("temp_max") < -2)
            .then(0)
            .otherwise(pl.col(colname))
            .alias(colname)
            for colname in [f"rain_hour_{hour:02d}" for hour in range(8, 19)]
        ]
    )

    # Join with visitor data
    data = data.join(weather_hourly, on="date")

    # Split into train and test sets
    data_train, data_test = (
        data.filter(
            (pl.col("date") < pl.date(2023, 1, 1))
            & (~pl.col("date").is_in(closed_dates.implode()))
        ),
        data.filter((pl.col("date") >= pl.date(2023, 1, 1))),
    )

    park_hours = list(range(8, 19))
    rain_columns = [f"rain_hour_{hour:02d}" for hour in park_hours]

    return data_test, data_train, park_hours, rain_columns, weekday_enum


@app.cell
def _(data_train):
    # Display sample of the training data
    data_train.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Define model

    We add a saturating effect on the hourly rainfall.
    The plot below shows the multiplicative effect of the rain, that is, 10mm per hour is a priori expected to reduce visits by 10% - 60%.
    """
    )
    return


@app.cell
def _():
    from pymc_marketing.prior import Prior
    from pymc_marketing.mmm.fourier import YearlyFourier

    # Yearly seasonality component
    yearly = YearlyFourier(
        n_order=4,
        prior=Prior("Laplace", mu=0, b=0.5, dims=("fourier", "weekdays")),
    )
    return Prior, yearly


@app.cell
def _(Prior, az, data_train, np, plt, rain_columns):
    from pymc_marketing.mmm.components.saturation import LogisticSaturation

    # Saturation function for hourly rain effects
    saturation_rain_hourly = LogisticSaturation(
        priors={
            "beta": Prior(
                "TruncatedNormal",
                mu=Prior("Normal", mu=-0.5, sigma=0.5),
                sigma=Prior("HalfNormal", sigma=0.5),
                upper=0,
                dims=("hours"),
            ),
            "lam": Prior("HalfNormal", sigma=2),
        },
        prefix="saturation_rain_hourly",
    )

    # Sample and plot the saturation curve
    _prior = saturation_rain_hourly.sample_prior(
        draws=1000, coords={"hours": range(8, 9)}
    )
    _curve = saturation_rain_hourly.sample_curve(_prior)
    _ax = az.plot_hdi(
        _curve.coords["x"] * data_train.select(rain_columns).to_numpy().max(),
        np.exp(_curve.sel(hours=8)),
    )
    az.plot_hdi(
        _curve.coords["x"] * data_train.select(rain_columns).to_numpy().max(),
        np.exp(_curve.sel(hours=8)),
        ax=_ax,
        hdi_prob=0.5,
    )
    plt.gca()
    return (saturation_rain_hourly,)


@app.cell
def _(
    data_train,
    park_hours,
    pm,
    pt,
    rain_columns,
    saturation_rain_hourly,
    weekday_enum,
    yearly,
):
    coords = {
        "date": data_train["date"],
        "weekdays": weekday_enum.categories,
        "hours": park_hours,  # range(8,19)
    }

    with pm.Model(coords=coords) as weather_hourly_model:
        # Fixed max values from training data
        _max_rain_hourly = data_train.select(rain_columns).to_numpy().max()
        _max_temperature = data_train["temp_avg"].max()

        # Input data
        _temperature = pm.Data("temperature", data_train["temp_avg"], dims=("date",))
        _day_of_year = pm.Data(
            "day_of_year", data_train["date"].dt.ordinal_day(), dims=("date",)
        )
        _day_of_week = pm.Data(
            "day_of_week", data_train["day_of_week"].to_physical(), dims=("date",)
        )
        _hourly_rain = pm.Data(
            "hourly_rain",
            data_train.select(rain_columns).to_numpy(),
            dims=("date", "hours"),
        )
        _obs_id = pt.arange(_day_of_year.shape[0])

        # Define parameters and their priors
        _avg_daily = pm.Normal("mu", mu=1000, sigma=300, dims=("weekdays",))
        _seasonality = pm.Deterministic(
            "seasonality", yearly.apply(_day_of_year), dims=("date", "weekdays")
        )
        _shape = pm.HalfNormal("shape", sigma=2, dims=("weekdays"))

        # _temperature_effect = pm.Normal("temperature_effect")
        _rain_contribution = pm.Deterministic(
            "rain_contribution",
            pt.sum(
                saturation_rain_hourly.apply(_hourly_rain / _max_rain_hourly),
                axis=1,
            ),
            dims=("date",),
        )

        pm.NegativeBinomial(
            "visitors",
            mu=_avg_daily[_day_of_week]
            + pt.exp(
                _seasonality[_obs_id, _day_of_week] + _rain_contribution
                #       + _temperature_effect * _temperature / _max_temperature
            ),
            alpha=_shape[_day_of_week],
            observed=data_train["visitors"],
            dims=("date",),
        )
    return (weather_hourly_model,)


@app.cell
def _(weather_hourly_model):
    weather_hourly_model
    return


@app.cell
def _(pm, weather_hourly_model):
    with weather_hourly_model:
        trace = pm.sample(nuts_sampler="nutpie")
        pm.sample_posterior_predictive(trace, extend_inferencedata=True)
    return (trace,)


@app.cell(hide_code=True)
def _(data_test, pm, rain_columns, trace, weather_hourly_model):
    with weather_hourly_model:
        pm.set_data(
            {
                "day_of_year": data_test["date"].dt.ordinal_day(),
                "day_of_week": data_test["day_of_week"].to_physical(),
                "hourly_rain": data_test.select(rain_columns).to_numpy(),
                "temperature": data_test["temp_avg"],
            },
            coords={"date": data_test["date"]},
        )
        predicted = pm.sample_posterior_predictive(trace, predictions=True).predictions
    return (predicted,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Inspect the results

    Main observation: big uncertainty on the rain effect parameters.
    """
    )
    return


@app.cell(hide_code=True)
def _(az, plt, trace):
    # Plot forest plot of key parameters
    az.plot_forest(
        trace,
        var_names=[
            "shape",
            "saturation_rain_hourly_lam",
            "saturation_rain_hourly_beta",
        ],
        combined=True,
    )
    plt.gca()
    return


@app.cell
def _(np, plt, saturation_rain_hourly, trace):
    _curve = saturation_rain_hourly.sample_curve(trace.posterior)
    saturation_rain_hourly.plot_curve(
        np.exp(_curve), subplot_kwargs={"figsize": (12, 3)}
    )
    plt.gca()
    return


@app.cell
def _(data_test, data_train, predicted, trace):
    from notebooks.model_baseline import plot_posterior_predictions

    plot_posterior_predictions(
        data_train, data_test, trace.posterior_predictive, predicted
    )
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
