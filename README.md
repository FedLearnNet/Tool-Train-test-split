# train-test-split

Splits a tabular dataset into training and test sets with configurable test size, shuffling, and random seed.

## Input and output

Provide a CSV table as `input`. The tool returns CSV tables named `train` and `test`.
All input columns are preserved, and each input row belongs to exactly one subset.

## Configuration

- `test_size`: test-set proportion, strictly between 0 and 1; default `0.25`.
- `shuffle`: shuffle rows before splitting; default `true`.
- `random_state`: seed used when shuffling; default `42`.

With `shuffle=false`, the training set contains the leading rows and the test set
contains the remaining rows. The Python config also accepts an integer `test_size`
for an absolute row count; the platform exposes the proportional FLOAT setting.

The split uses scikit-learn's `train_test_split`.

`data/example.csv` is a small, generated synthetic dataset for local examples.

## Development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
# Set APP_ID and platform URLs in .env before connecting.
python main.py
```

The Docker base image provides the FL-Net API. `requirements.txt` contains only
additional runtime dependencies; `requirements-dev.txt` includes the API for local development.

Build with `docker build -t tool-train-test-split .`.
