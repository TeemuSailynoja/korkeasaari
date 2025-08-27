import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Zoo model: events and weather""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Data imports """)
    return


@app.cell
def _():
    # Events PyMC model: intercept + seasonal effect per weekday + weather effects with saturation + special events
    from datetime import date
    import polars as pl
    import numpy as np
    import pymc as pm
    import pytensor.tensor as pt
    import matplotlib.pyplot as plt
    import arviz as az
    import marimo as mo

    return az, date, mo, np, pl, plt, pm, pt


@app.cell
def load_data(date, events, pl, vacations):
    from notebooks.model_weather_hourly import load_data

    # use the data loading from the base model notebook
    _dl_res = load_data.run()[1]
    data = _dl_res["data"]
    weekday_enum = _dl_res["weekday_enum"]
    closed_dates = _dl_res["closed_dates"]
    # encode weekdays
    data = (
        data.join(events, "date", "left")
        .join(vacations, "date", "left")
        .rename({"Vacation": "vacation"})
    )
    data = data.with_columns(
        pl.col("name")
        .fill_null("No Event")
        .cast(pl.Enum(["No Event"] + list(data["name"].unique().drop_nulls()))),
        pl.col("vacation")
        .fill_null("No Vacation")
        .cast(pl.Enum(["No Vacation"] + list(data["vacation"].unique().drop_nulls()))),
        pl.when(pl.col("date") == date(2022, 9, 2))
        .then(1)
        .otherwise(0)
        .alias("tiger_cubs"),
    )
    data_train, data_test = (
        data.filter(
            (pl.col("date") < pl.date(2023, 1, 1))
            & (~pl.col("date").is_in(closed_dates.implode()))
        ),
        data.filter((pl.col("date") >= pl.date(2023, 1, 1))),
    )

    park_hours = list(range(8, 19))
    rain_columns = [f"rain_hour_{hour:02d}" for hour in park_hours]

    return data, data_test, data_train, park_hours, rain_columns, weekday_enum


@app.cell
def _(pl):
    events = (
        pl.read_csv("data/clean/all_events.csv")
        .with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d").alias("date"))
        .unique()
    )
    vacations = (
        pl.read_csv("data/clean/finnish_school_holidays_daily_2018_2024.csv")
        .rename({"Date": "date"})
        .with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d").alias("date"))
    )
    events.head()
    return events, vacations


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Define the model
    Fun fun!
    """
    )
    return


@app.cell
def _():
    from pymc_marketing.prior import Prior
    from pymc_marketing.mmm.fourier import YearlyFourier

    # sigma_yearly = YearlyFourier(n_order=1, prefix="fourier_sigma",prior = Prior("Normal",mu = 0, sigma = .2))
    yearly = YearlyFourier(
        n_order=6,
        prior=Prior("Laplace", mu=0, b=0.5, dims=("fourier", "weekdays")),
    )

    yearly_sd = YearlyFourier(
        n_order=4,
        prior=Prior("Laplace", mu=0, b=0.5, dims=("seasonal_sigma", "weekdays")),
        prefix="seasonal_sigma",
    )
    return Prior, yearly, yearly_sd


@app.cell
def _(Prior):
    from pymc_marketing.mmm.components.saturation import LogisticSaturation

    saturation_rain = LogisticSaturation(
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
        prefix="saturation_rain",
    )
    return (saturation_rain,)


@app.cell
def _(
    data,
    data_train,
    park_hours,
    pm,
    pt,
    rain_columns,
    saturation_rain,
    weekday_enum,
    yearly,
    yearly_sd,
):
    coords = {
        "date": data_train["date"],
        "weekdays": weekday_enum.categories,
        "hours": park_hours,
        "events": [event for event in data["name"].unique() if event != "No Event"],
        "vacations": [
            vacation
            for vacation in data["vacation"].unique()
            if vacation != "No Vacation"
        ],
    }

    with pm.Model(coords=coords) as event_model:
        _max_rain = data_train.select(rain_columns).to_numpy().max()

        # Define input data
        _day_of_year = pm.Data(
            "day_of_year", data_train["date"].dt.ordinal_day(), dims=("date",)
        )
        _day_of_week = pm.Data(
            "day_of_week", data_train["day_of_week"].to_physical(), dims=("date",)
        )
        _rain = pm.Data(
            "rain",
            data_train.select(rain_columns).to_numpy(),
            dims=("date", "hours"),
        )
        _obs_id = pt.arange(_day_of_year.shape[0])

        ## New inputs
        _tiger_cubs = pm.Data("tiger_cubs", data_train["tiger_cubs"], dims=("date",))

        _events = pm.Data("events", data_train["name"].to_physical(), dims=("date",))
        _vacations = pm.Data(
            "vacations", data_train["vacation"].to_physical(), dims=("date",)
        )

        # Define parameters and their priors
        _avg_daily = pm.Normal("mu", mu=1000, sigma=200, dims=("weekdays"))

        _seasonality = pm.Deterministic(
            "seasonality", yearly.apply(_day_of_year), dims=("date", "weekdays")
        )
        _shape = pm.HalfNormal("shape", sigma=2, dims=("weekdays"))

        _seasonal_shape = pm.Deterministic(
            "shape_seasonal", yearly_sd.apply(_day_of_year), dims=("date", "weekdays")
        )
        _event_effects = pm.TruncatedNormal(
            "event_effect", mu=0, sigma=1, lower=0, dims=("events",)
        )

        _peak_tiger_hype = pm.Normal("tiger_hype_peak", mu=0.1, sigma=0.1)
        _delay_tiger_hype = pm.TruncatedNormal(
            "tiger_hype_delay", mu=30, sigma=10, lower=1
        )
        _decay_tiger_hype = pm.TruncatedNormal(
            "tiger_hype_decay", mu=30, sigma=10, lower=7
        )

        _tiger_hype = pm.Deterministic(
            "tiger_hype",
            _tiger_cubs.cumsum()
            * _peak_tiger_hype
            * pm.math.exp(
                -0.5
                * ((_tiger_cubs.cumsum().cumsum() - _delay_tiger_hype) ** 2)
                / _decay_tiger_hype**2
            ),
            dims=("date",),
        )

        _vacation_effects = pm.Normal(
            "vacation_effect",
            mu=pm.Normal("vacation_mu"),
            sigma=pm.HalfNormal("vacation_sigma"),
            dims=("vacations",),
        )

        _event_contribution = pm.Deterministic(
            "event_contribution",
            pt.pad(_event_effects, (1, 0), constant_values=(0, 0), mode="constant")[
                _events
            ],
            dims=("date",),
        )

        _vacation_contribution = pm.Deterministic(
            "vacation_contribution",
            pt.pad(_vacation_effects, (1, 0), constant_values=(0, 0), mode="constant")[
                _vacations
            ],
            dims=("date",),
        )

        _rain_contribution = pm.Deterministic(
            "rain_contribution",
            pt.sum(saturation_rain.apply(_rain / _max_rain), axis=1),
            dims=("date",),
        )

        pm.NegativeBinomial(
            "visitors",
            mu=_avg_daily[_day_of_week]
            * pt.exp(
                _seasonality[_obs_id, _day_of_week]
                + _rain_contribution
                + _event_contribution
                + _vacation_contribution
                + _tiger_hype
            ),
            alpha=_shape[_day_of_week] * pt.exp(_seasonal_shape[_obs_id, _day_of_week]),
            observed=data_train["visitors"],
            dims=("date",),
        )
    return (event_model,)


@app.cell
def _(event_model):
    event_model
    return


@app.cell(hide_code=True)
def _(event_model, pm):
    with event_model:
        trace = pm.sample(nuts_sampler="nutpie", target_accept=0.95)
        pm.sample_posterior_predictive(trace, extend_inferencedata=True)
    return (trace,)


@app.cell(hide_code=True)
def _(data, event_model, park_hours, pm, trace):
    with event_model:
        pm.set_data(
            {
                "day_of_year": data["date"].dt.ordinal_day(),
                "day_of_week": data["day_of_week"].to_physical(),
                "rain": data[[f"rain_hour_{hour:02d}" for hour in park_hours]],
                "events": data["name"].to_physical(),
                "vacations": data["vacation"].to_physical(),
                "tiger_cubs": data["tiger_cubs"].to_physical(),
            },
            coords={"date": data["date"]},
        )
        predicted = pm.sample_posterior_predictive(
            trace,
            predictions=True,
            var_names=["visitors", "rain_contribution", "tiger_hype"],
            extend_inferencedata=False,
        ).predictions
    posterior_sampled = True
    return posterior_sampled, predicted


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Look at the results""")
    return


@app.cell
def _(az, posterior_sampled, trace):
    posterior_sampled
    az.plot_forest(
        trace.posterior,
        combined=True,
        var_names=[
            "tiger_hype_peak",
            "event_effect",
            "vacation_effect",
            "vacation_mu",
            "vacation_sigma",
            "saturation_rain_beta",
        ],
    )
    return


@app.cell
def _(az, posterior_sampled, trace):
    posterior_sampled
    az.summary(
        trace.posterior,
        var_names=["tiger_hype_peak", "tiger_hype_delay", "tiger_hype_decay"],
    )
    return


@app.cell
def _(az, data_train, np, plt, trace):
    _fig, _ax = plt.subplots(nrows=7, figsize=(14, 12))

    for _idx, _weekday in enumerate(data_train["day_of_week"].unique()):
        az.plot_hdi(
            data_train["date"],
            trace.posterior["mu"]
            .sel({"weekdays": _weekday})
            .expand_dims(dim={"date": data_train["date"].to_list()})
            .transpose("chain", "draw", "date")
            * np.exp(
                trace.posterior["seasonality"].sel({"weekdays": _weekday})
                + trace.posterior["vacation_contribution"]
                + trace.posterior["event_contribution"]
            ),
            smooth=False,
            ax=_ax[_idx],
        )
        _ax[_idx].set_title(_weekday)

    _fig
    return


@app.cell
def _(data_test, data_train, predicted, trace):
    from notebooks.model_baseline import plot_posterior_predictions

    plot_posterior_predictions(
        data_train,
        data_test,
        trace.posterior_predictive,
        predicted.sel(date=data_test["date"]),
    )
    return


@app.cell
def _(data_test, data_train, predicted, trace):
    from notebooks.model_baseline import plot_errors

    plot_errors(
        trace.posterior_predictive["visitors"],
        data_train,
        predicted.sel(date=data_test["date"])["visitors"],
        data_test,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Effects of rain and tigers!""")
    return


@app.cell
def _(az, data_test, np, predicted):
    rain_effect = (
        predicted["visitors"] * (1 - 1 / np.exp(predicted["rain_contribution"]))
    ).sel(date=data_test["date"])
    _ax = az.plot_hdi(
        data_test["date"],
        rain_effect,
        smooth=False,
        hdi_prob=0.5,
        color="#81a1c1",
    )
    az.plot_hdi(
        data_test["date"],
        rain_effect,
        smooth=False,
        ax=_ax,
        hdi_prob=0.1,
        color="#81a1c1",
    )
    return (rain_effect,)


@app.cell
def _(az, data_test, rain_effect):
    _ax = az.plot_hdi(
        data_test["date"],
        rain_effect.cumsum("date"),
        smooth=False,
        hdi_prob=0.9,
        color="#81a1c1",
    )
    az.plot_hdi(
        data_test["date"],
        rain_effect.cumsum("date"),
        smooth=False,
        hdi_prob=0.5,
        color="#81a1c1",
    )
    return


@app.cell
def _(az, data, np, predicted):
    full_hype = predicted["visitors"] * (1 - 1 / np.exp(predicted["tiger_hype"]))
    _ax = az.plot_hdi(
        data["date"], full_hype, smooth=False, hdi_prob=0.8, color="#81a1c1"
    )
    _ax = az.plot_hdi(
        data["date"], full_hype, smooth=False, hdi_prob=0.5, color="#81a1c1"
    )
    az.plot_hdi(
        data["date"],
        full_hype,
        smooth=False,
        ax=_ax,
        hdi_prob=0.1,
        color="#81a1c1",
        fill_kwargs={"alpha": 1},
    )
    return (full_hype,)


@app.cell
def _(az, data, full_hype):
    _ax = az.plot_hdi(
        data["date"],
        full_hype.cumsum("date"),
        smooth=False,
        hdi_prob=0.8,
        color="#81a1c1",
    )
    az.plot_hdi(
        data["date"],
        full_hype.cumsum("date"),
        smooth=False,
        ax=_ax,
        hdi_prob=0.5,
        color="#81a1c1",
    )
    return


if __name__ == "__main__":
    app.run()
