# groundrails

Grounding guardrails for agentic RAG - deterministic claim verification

> **Note**: Generated with copier-data-science template v1.2+
> For template documentation, visit [copier-data-science](https://github.com/stellarshenson/copier-data-science)

## Quick Start

```bash
make install
```

## Makefile Targets

- `make install` - Create environment and install package
- `make test` - Run tests
- `make lint` / `make format` - Check / fix code style
- `make build` - Build distributable wheel
- `make clean` - Remove compiled files and caches
- `make sync_data_down` / `make sync_data_up` - Sync data with cloud storage
- `make sync_models_down` / `make sync_models_up` - Sync models with cloud storage
- `make .env` / `make .env.enc` - Decrypt / encrypt environment secrets
- `make help` - Show all available targets

## Best Practices

- **Notebooks**: Name with number prefix, initials, description - `01-jqp-data-exploration.ipynb`
- **Data**: Keep `raw/` immutable, use `interim/` for transforms, `processed/` for final datasets
- **Source code**: Refactor reusable notebook code into `src/groundrails/` modules
- **Models**: Store trained models in `models/` with clear naming

## Project Organization

```
├── Makefile           <- Makefile with convenience commands
├── README.md          <- The top-level README for developers
├── data
│   ├── external       <- Data from third party sources
│   ├── interim        <- Intermediate data that has been transformed
│   ├── processed      <- The final, canonical data sets for modeling
│   └── raw            <- The original, immutable data dump
│
├── models             <- Trained and serialized models
├── notebooks          <- Jupyter notebooks
├── pyproject.toml     <- Project configuration and dependencies
├── references         <- Data dictionaries, manuals, explanatory materials
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures
├── tests              <- Test files
└── src
    └── groundrails   <- Source code for this project
        ├── __init__.py
        ├── config.py      <- Configuration variables
        ├── dataset.py     <- Data download/generation scripts
        ├── features.py    <- Feature engineering code
        ├── modeling
        │   ├── predict.py <- Model inference
        │   └── train.py   <- Model training
        └── plots.py       <- Visualization code
```
