# Random Forest with a PostgreSQL Data Pipeline

Recognizes handwritten digits from 8x8 images using **Python, scikit-learn, and
NumPy**. **SQLAlchemy + PostgreSQL** store samples, training runs, and predictions.
SQL selects train/test data and joins predictions to labels for error reports.
The saved model scored 96.4% on 450 test images, versus a 10.2% baseline.

## Database demo with Docker

From this folder, with Docker running:

```bash
docker compose up --build --abort-on-container-exit --exit-code-from app app
docker compose --profile test run --rm tests
docker compose exec -T db psql -U student -d digits -f /sql/report.sql
docker compose down
```

The app imports 1,797 samples, trains from database rows, and saves 450 test
predictions. Repeated imports preserve existing rows; each training run gets a
new ID and a separate model under `artifacts/runs/`. `artifacts/database_demo.json`
records actual results. SQL reports show class balance, model runs, and errors by
digit. PostgreSQL data survives in a Docker volume. Start it again with
`docker compose up -d db` before running SQL reports or the database tests.
Demo credentials are local examples; the database has no published host port.

## Python and standalone model

Use Python 3.12:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python digits.py train
python digits.py predict examples/digit.csv
python -m pytest -q
```

On Windows, activate `.venv\Scripts\activate`. The PostgreSQL integration test is
skipped in a plain local run; the Compose test command runs it against PostgreSQL
in an isolated test schema. Other data-layer tests use SQLite. All tests run
during image builds except that integration check.

For an existing database, set `DATABASE_URL` (for example a
`postgresql+psycopg://user:password@host:5432/digits` URL) and run
`python digits.py pipeline`. Individual commands are `db-seed`, `db-train`, and
`db-report`. The import fixes the dataset split; `--seed` controls the import split
and forest seed. Changing the import seed requires a fresh dataset/database.
Model and dataset SHA-256 hashes identify each saved run.

The database layer is separated into `database.py`; SQLAlchemy Core keeps queries
straightforward to adapt. PostgreSQL and SQLite are tested. A Snowflake backend
would need its own connector, schema adaptation, and integration tests.

Data: [scikit-learn digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html),
adapted from UCI Optical Recognition of Handwritten Digits. Images must be centered
8x8 grids with values 0–16; this is not a phone-photo recognizer. Vote share is not
a calibrated probability. Only load joblib model files you trust.
