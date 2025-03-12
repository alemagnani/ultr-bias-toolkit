# ULTR-BIAS-TOOLKIT DEVELOPMENT GUIDE

## Commands
- Install dependencies: `poetry install`
- Run tests: `poetry run pytest`
- Run single test: `poetry run pytest path/to/test.py::test_function_name -v`
- Lint: `poetry run black .` (install with `poetry add black --dev`)
- Type check: `poetry run mypy .` (install with `poetry add mypy --dev`)

## Code Style
- **Imports**: Standard library first, third-party next, local modules last, all alphabetical
- **Type Hints**: Use type hints for function parameters and return values
- **Naming**:
  - Classes: PascalCase (e.g., `NaiveCtrEstimator`)
  - Functions/methods: snake_case (e.g., `build_intervention_sets`)
  - Variables: snake_case (e.g., `query_col`, `doc_col`)
- **Error Handling**: Use assertions with descriptive error messages
- **Documentation**: Docstrings for classes and methods using Google style
- **Dataframes**: Use `assert_columns_in_df` to validate required DataFrame columns

## Input Formats
The toolkit supports two input formats for click data:
1. **Binary format**: Each row represents a single impression with `click=0/1`
2. **Aggregated format**: Rows contain `impressions` and `clicks` counts for efficiency with large datasets

When using aggregated format, always ensure that clicks <= impressions. Pass the column names via the `imps_col` and `clicks_col` parameters.