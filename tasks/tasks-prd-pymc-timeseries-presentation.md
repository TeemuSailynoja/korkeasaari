
## Relevant Files

- `notebooks/marketing_model_pymc.ipynb` - Contains modeling workflow and code for presentation.
- `notebooks/parse_and_load_data.ipynb` - Data loading and preprocessing steps.
- `data/clean/visitors.csv` - Cleaned daily visitor data.
- `data/clean/weather.csv` - Cleaned weather data.
- `korkeasaari/model.py` - Main PyMC model logic and incremental improvements.
- `korkeasaari/utils.py` - Utility functions for data handling and plotting.
- `notebooks/parse_weather_data.ipynb` - Weather data parsing and cleaning.
- `presentation/nord.scss` - Custom Nord color theme for Quarto revealjs presentation.



### Notes

All dependencies should be managed using pixi (see `pixi.lock` and `pyproject.toml`).

## Tasks

- [x] 1.0 Set up Quarto revealjs presentation with Nord color scheme
  - [x] 1.1 Create a new Quarto project using revealjs format
  - [x] 1.2 Configure Nord color palette for backgrounds, text, and plots
  - [x] 1.3 Set up slide structure for incremental modeling steps
  - [x] 1.4 Test export to HTML and check visual consistency

- [ ] 2.0 Prepare and load Korkeasaari Zoo daily visitor and weather data
  - [x] 2.1 Review and clean `visitors.csv` and `weather.csv` in `data/clean/`
  - [x] 2.2 Use polars to load and preprocess data in notebook/code
  - [x] 2.3 Document data sources and preprocessing steps in presentation
  - [x] 2.4 Prepare for easy addition of holidays/events data

- [ ] 3.0 Implement and demonstrate incremental PyMC timeseries models
  - [ ] 3.1 Implement baseline model: intercept + seasonal effect
  - [ ] 3.2 Add weather predictors to the model
  - [ ] 3.3 Add holidays/events predictors (placeholder for future data)
  - [ ] 3.4 Check model convergence and compare results at each step
  - [ ] 3.5 Document code and modeling logic for each step

- [ ] 4.0 Create visualizations and code explanations for each modeling step
  - [ ] 4.1 Generate time series plots for observed and predicted values
  - [ ] 4.2 Visualize additive term effects and model comparisons
  - [ ] 4.3 Highlight and explain code snippets using revealjs features
  - [ ] 4.4 Summarize convergence checks and model improvements

- [ ] 5.0 Finalize presentation, export formats, and appendix with parameter posteriors
  - [ ] 5.1 Add appendix with parameter posterior plots
  - [ ] 5.2 Ensure presentation is reproducible from provided data/code
  - [ ] 5.3 Export final presentation as HTML and (optionally) Marimo notebook
  - [ ] 5.4 Review for clarity, accessibility, and visual consistency
