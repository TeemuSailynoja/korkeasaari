# Product Requirements Document: PyMC Timeseries Model Presentation for PyData

## 1. Introduction/Overview
This project aims to create a Quarto revealjs presentation (with optional Marimo/Jupyter notebook version) for PyData attendees, demonstrating the incremental improvement of a PyMC timeseries model using Korkeasaari Zoo daily visitor data. The presentation will highlight the Bayesian modeling workflow, starting from a simple model and gradually incorporating weather and special events, with a focus on clarity and accessibility for those new to PyMC and Bayesian methods.

## 2. Goals
- Deliver a clear, engaging 30–45 minute presentation for PyData attendees
- Demonstrate the process of incrementally improving a Bayesian timeseries model
- Use real-world data (visitor counts, weather, holidays/events)
- Provide code explanations and visualizations at each modeling step
- Follow the Nord color scheme for visual consistency

## 3. User Stories
- As a PyData attendee, I want to see how a simple model can be improved step-by-step so I can understand the value of Bayesian modeling.
- As a presenter, I want to highlight code and show scrollable snippets so the audience can follow the logic without being overwhelmed.
- As a data science learner, I want to see the impact of adding predictors (weather, events) so I can appreciate model interpretability.

## 4. Functional Requirements
1. The presentation must be created in Quarto using the revealjs format, with Nord color scheme styling.
2. The presentation must load daily visitor data from `data/clean/visitors.csv` and, in later steps, weather data from `data/clean/weather.csv`.
3. The presentation must demonstrate three main modeling steps:
   1. Baseline model: intercept + seasonal effect
   2. Add weather predictors
   3. Add holidays/events (to be provided)
4. Each modeling step must include:
   - Code explanations (with highlighted/scrollable code snippets)
   - Visualizations: time series plots, additive term effects, and model comparison plots
   - Brief summary of convergence checks
   - Comparison to previous step
5. The appendix must include parameter posterior plots.
6. The presentation must be exportable as HTML and optionally as a Marimo/Jupyter notebook.
7. The presentation must be suitable for a human presenter (not fully automated narration).
8. The presentation must be visually consistent and accessible, following the Nord color scheme.

## 5. Non-Goals (Out of Scope)
- Deep technical dives into PyMC internals or advanced Bayesian theory
- Use of unrelated datasets
- Fully automated or narrated presentation
- In-depth focus on parameter posteriors (beyond appendix)

## 6. Design Considerations
- Use the Nord color palette for all plots and backgrounds
- Leverage Quarto revealjs features for code highlighting and scrollable code blocks
- Ensure all visualizations are clear and not overly technical
- Provide enough material for a 30–45 minute talk

## 7. Technical Considerations
- Presentation should be reproducible from the provided data and code
- Use PyMC for modeling, pandas for data handling, and matplotlib/seaborn for plots
- Ensure compatibility with Quarto and Marimo/Jupyter
- Holidays/events data will be provided later and should be easy to incorporate

## 8. Success Metrics
- Presentation is delivered smoothly within 30–45 minutes
- Audience feedback indicates improved understanding of Bayesian modeling
- All code and visualizations are clear and reproducible
- Visual style is consistent and appealing

## 9. Open Questions
- When will the list of holidays/events be provided?
- Should the presentation include a downloadable notebook version for attendees?
- Are there any accessibility requirements (e.g., font size, color contrast)?
