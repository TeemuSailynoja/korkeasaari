import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell
def _():
    # Weekdays PyMC model: intercept + seasonal effect per weekday
    import polars as pl
    import numpy as np
    import pymc as pm
    import pytensor.tensor as pt
    import matplotlib.pyplot as plt
    import xarray as xr

    return np, pl, plt, pm, pt, xr


@app.cell
def load_data(pl):
    from notebooks.model_baseline import load_data

    # use the data loading from the base model notebook
    _dl_res = load_data.run()[1]
    data = _dl_res["data"]
    closed_dates = _dl_res["closed_dates"]

    # encode weekdays
    weekday_enum = pl.Enum(["ma", "ti", "ke", "to", "pe", "la", "su"])
    data = data.with_columns(pl.col("day_of_week").cast(weekday_enum)).join(
        pl.read_csv("data/clean/helsinki_kaisaniemi_daily_filtered.csv")
        .with_columns(
            temp=pl.col("Lämpötilan keskiarvo [°C]"),
            rain=pl.col("Tunnin sademäärä [mm]"),
            date=pl.col("date").str.to_date("%Y-%m-%d"),
        )
        .select(["temp", "rain", "date"]),
        on="date",
    )
    data = data.with_columns(
        pl.when(pl.col("temp") < -2).then(0).otherwise(pl.col("rain")).alias("rain")
    )

    data_train, data_test = (
        data.filter(
            (pl.col("date") < pl.date(2023, 1, 1))
            & (~pl.col("date").is_in(closed_dates.implode()))
        ),
        data.filter((pl.col("date") >= pl.date(2023, 1, 1))),
    )
    data_train.head()
    return data_test, data_train, weekday_enum


@app.cell
def _():
    from pymc_marketing.prior import Prior
    from pymc_marketing.mmm.fourier import YearlyFourier

    yearly = YearlyFourier(
        n_order=4,
        prior=Prior("Laplace", mu=0, b=0.5, dims=("fourier", "weekdays")),
    )
    return (yearly,)


@app.cell
def _(data_train, pm, pt, weekday_enum, yearly):
    coords = {"date": data_train["date"], "weekdays": weekday_enum.categories}

    with pm.Model(coords=coords) as weather_model:
        # Define scalers that will not change when data is changed
        _max_visitors = data_train["visitors"].max()
        _max_rain = data_train["rain"].max()
        _max_temperature = data_train["temp"].max()

        # Define data that can be changed when predicting
        _day_of_year = pm.Data(
            "day_of_year", data_train["date"].dt.ordinal_day(), dims=("date",)
        )
        _day_of_week = pm.Data(
            "day_of_week", data_train["day_of_week"].to_physical(), dims=("date",)
        )
        _rain = pm.Data("rain", data_train["rain"], dims=("date",))
        _temperature = pm.Data("temperature", data_train["temp"], dims=("date",))
        _obs_id = pt.arange(_day_of_year.shape[0])

        # Define model parameters and their priors
        _avg_daily = pm.Normal("mu", mu=1000, sigma=200, dims=("weekdays"))
        _seasonality = pm.Deterministic(
            "seasonality", yearly.apply(_day_of_year), dims=("date", "weekdays")
        )
        _temperature_effect = pm.Normal("temperature_effect")
        _rain_effect = pm.Normal("rain_effect")
        _daily_sd = pm.HalfNormal("sigma", sigma=2, dims=("weekdays"))

        # Model on a 0-1 scale for numerical stability.
        _visitors_scaled = pm.NegativeBinomial(
            "visitors_predicted",
            mu=_avg_daily[_day_of_week]
            * pt.exp(
                _seasonality[_obs_id, _day_of_week]
                + _rain_effect * _rain / _max_rain
                + _temperature_effect * _temperature / _max_temperature
            ),
            alpha=_daily_sd[_day_of_week],
            observed=data_train["visitors"],
            dims=("date",),
        )

        # Scale back to visitors
        # pm.Deterministic(
        #     "visitors_predicted",
        #     _visitors_scaled * _max_visitors,
        #     dims=("date",)
        # )
    return (weather_model,)


@app.cell
def _(weather_model):
    weather_model
    return


@app.cell
def _(data_test, pm, weather_model):
    with weather_model:
        trace = pm.sample(nuts_sampler="nutpie")
        pm.sample_posterior_predictive(trace, extend_inferencedata=True)
        pm.set_data(
            {
                "day_of_year": data_test["date"].dt.ordinal_day(),
                "day_of_week": data_test["day_of_week"].to_physical(),
                "rain": data_test["rain"],
                "temperature": data_test["temp"],
            },
            coords={"date": data_test["date"]},
        )
        predicted = pm.sample_posterior_predictive(trace, predictions=True).predictions
    return predicted, trace


@app.cell
def _():
    import arviz as az

    return (az,)


@app.cell
def _():
    return


@app.cell
def _(az, np, trace):
    az.plot_forest(
        data=np.exp(trace.posterior[["temperature_effect", "rain_effect"]]).assign(
            sigma=trace.posterior["sigma"]
        ),
        var_names=["temperature_effect", "rain_effect", "sigma"],
        combined=True,
    )
    return


@app.cell
def _(az, trace):
    az.plot_forest(data=trace.posterior, var_names=["mu"], combined=True)
    return


@app.cell
def _(az, data_test, data_train, plt, predicted, trace):
    _fig, _ax = plt.subplots()
    _ax = az.plot_hdi(
        data_train["date"],
        y=trace.posterior_predictive["visitors_predicted"],
        smooth=False,
        hdi_prob=0.5,
        color="#81a1c1",
        fill_kwargs={"alpha": 1},
    )
    az.plot_hdi(
        data_train["date"],
        y=trace.posterior_predictive["visitors_predicted"],
        smooth=False,
        ax=_ax,
        color="#81a1c1",
    )
    az.plot_hdi(
        data_test["date"],
        y=predicted["visitors_predicted"],
        smooth=False,
        hdi_prob=0.5,
        ax=_ax,
        color="#88c0d0",
        fill_kwargs={"alpha": 1},
    )
    az.plot_hdi(
        data_test["date"],
        y=predicted["visitors_predicted"],
        smooth=False,
        ax=_ax,
        color="#88c0d0",
    )
    _ax.scatter(data_train["date"], data_train["visitors"], s=0.8, c="#bf616a")
    _ax.scatter(data_test["date"], data_test["visitors"], s=0.8, c="#d08770")
    return


@app.cell
def _(az, data_test, data_train, predicted, trace, xr):
    from notebooks.model_baseline import compute_errors

    visitors_train_xr = xr.DataArray(
        data_train["visitors"], coords={"date": data_train["date"].to_list()}
    )
    visitors_test_xr = xr.DataArray(
        data_test["visitors"], coords={"date": data_test["date"].to_list()}
    )

    az.plot_posterior(
        compute_errors(
            trace.posterior_predictive["visitors_predicted"],
            visitors_train_xr,
            predicted["visitors_predicted"],
            visitors_test_xr,
        ),
        grid=(2, 2),
        figsize=(10, 8),
    )
    return


if __name__ == "__main__":
    app.run()
