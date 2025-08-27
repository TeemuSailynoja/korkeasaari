import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell
def _():
    # Weather2 PyMC model: intercept + seasonal effect per weekday + weather effects with saturation
    import numpy as np
    import pymc as pm
    import pytensor.tensor as pt
    import matplotlib.pyplot as plt
    import xarray as xr

    return np, plt, pm, pt, xr


@app.cell
def load_data():
    from notebooks.model_weather import load_data

    # use the data loading from the base model notebook
    _dl_res = load_data.run()[1]
    data_train = _dl_res["data_train"]
    data_test = _dl_res["data_test"]
    weekday_enum = _dl_res["weekday_enum"]

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
    return Prior, yearly


@app.cell
def _(Prior, az, np):
    from pymc_marketing.mmm.components.saturation import LogisticSaturation

    saturation_rain = LogisticSaturation(
        priors={
            "lam": Prior("HalfNormal", sigma=5),
            "beta": Prior("TruncatedNormal", mu=-0.5, sigma=0.5, upper=0),
        },
        prefix="saturation_rain",
    )
    _prior = saturation_rain.sample_prior(draws=1000)
    _curve = saturation_rain.sample_curve(_prior)
    _ax = az.plot_hdi(_curve.coords["x"], np.exp(_curve))
    az.plot_hdi(_curve.coords["x"], np.exp(_curve), ax=_ax, hdi_prob=0.5)
    # _fig, _ax = saturation_rain.plot_curve(_curve)
    # saturation_rain.plot_curve_hdi(_curve, {"hdi_prob" : .5}, axes=_ax)[0]
    return (saturation_rain,)


@app.cell
def _(data_train, pm, pt, saturation_rain, weekday_enum, yearly):
    coords = {
        "date": data_train["date"],
        "weekdays": weekday_enum.categories,
    }

    with pm.Model(coords=coords) as weather_model:
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
        _avg_daily = pm.Normal("mu", mu=1000, sigma=300, dims=("weekdays"))
        _seasonality = pm.Deterministic(
            "seasonality", yearly.apply(_day_of_year), dims=("date", "weekdays")
        )
        _temperature_effect = pm.Normal("temperature_effect")
        _daily_sd = pm.HalfNormal("sigma", sigma=2, dims=("weekdays",))

        _rain_contribution = pm.Deterministic(
            "rain_contribution",
            saturation_rain.apply(_rain / _max_rain),
            dims=("date"),
        )
        # Model on a 0-1 scale for numerical stability.
        _visitors_scaled = pm.NegativeBinomial(
            "visitors_predicted",
            mu=_avg_daily[_day_of_week]
            + pt.exp(
                _seasonality[_obs_id, _day_of_week]
                + _rain_contribution
                + _temperature_effect * _temperature / _max_temperature
            ),
            # lower=0,
            alpha=_daily_sd[_day_of_week],
            observed=data_train["visitors"],
            dims=("date",),
        )
    return (weather_model,)


@app.cell
def _(weather_model):
    weather_model
    return


@app.cell
def _(pm, weather_model):
    with weather_model:
        trace = pm.sample(nuts_sampler="nutpie")
        pm.sample_posterior_predictive(trace, extend_inferencedata=True)
    return (trace,)


@app.cell
def _(data_test, pm, trace, weather_model):
    with weather_model:
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
    return (predicted,)


@app.cell
def _(az, trace):
    az.plot_forest(
        trace.posterior,
        var_names=[
            "sigma",
            "saturation_rain_lam",
            "saturation_rain_beta",
            "temperature_effect",
        ],
        combined=True,
    )
    return


@app.cell
def _(az, trace):
    az.summary(
        trace,
        var_names=[
            "mu",
            "sigma",
            "saturation_rain_lam",
            "saturation_rain_beta",
            # "saturation_rain_sigma",
        ],
    )
    return


@app.cell
def _(az, data_train, np, trace):
    _ax = az.plot_hdi(
        data_train["date"],
        y=np.exp(trace.posterior["rain_contribution"]),
        smooth=False,
        hdi_prob=0.5,
        color="#81a1c1",
    )
    az.plot_hdi(
        data_train["date"],
        y=np.exp(trace.posterior["rain_contribution"]),
        smooth=False,
        hdi_prob=0.94,
        color="#81a1c1",
        ax=_ax,
    )
    return


@app.cell
def _():
    import arviz as az

    return (az,)


@app.cell
def _(az, data_test, data_train, plt, predicted, trace):
    _fig, _ax = plt.subplots()
    _ax = az.plot_hdi(
        data_train["date"],
        y=trace.posterior_predictive["visitors_predicted"],
        smooth=False,
        hdi_prob=0.5,
        color="#81a1c1",
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
    )
    return


if __name__ == "__main__":
    app.run()
