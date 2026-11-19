import json
import os
from uuid import uuid4

import numpy as np
import pytest
from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.schema import CreateSchema, DropSchema

from database import connect, load_data, metadata, predictions, samples, save_run, seed_database, summary
from digits import split_data, train_database


@pytest.fixture
def engine():
    db = connect('sqlite://')
    yield db
    db.dispose()


def test_import_is_idempotent_and_preserves_original_split(engine):
    assert seed_database(engine)
    assert not seed_database(engine)
    data, test_rows, digest = load_data(engine)
    for original, stored in zip(split_data(), data):
        np.testing.assert_array_equal(original, stored)
    assert len(test_rows) == 450 and len(digest) == 64
    assert summary(engine)['row_counts']['samples'] == 1797


def test_import_refuses_to_overwrite_a_different_split(engine):
    seed_database(engine)
    with pytest.raises(ValueError, match='differ'):
        seed_database(engine, seed=99)
    assert summary(engine)['row_counts']['samples'] == 1797


def test_sql_constraints_reject_invalid_labels_and_relations(engine):
    metadata.create_all(engine)
    with pytest.raises(IntegrityError), engine.begin() as conn:
        conn.execute(samples.insert(), {'id': 1, 'label': 11, 'split': 'train',
                                       'split_order': 0, 'pixels_json': '[]'})
    with pytest.raises(IntegrityError), engine.begin() as conn:
        conn.execute(predictions.insert(), {'run_id': 'missing', 'sample_id': 123,
                                           'predicted_label': 1, 'vote_share': .5})


def test_invalid_database_pixels_are_rejected(engine):
    seed_database(engine)
    with engine.begin() as conn:
        conn.execute(update(samples).where(samples.c.id == 1).values(pixels_json='[99]'))
    with pytest.raises(ValueError, match='64 finite pixel'):
        load_data(engine)


def test_failed_prediction_insert_rolls_back_the_run(engine):
    seed_database(engine)
    metrics = {'seed': 42, 'accuracy': 1.0, 'macro_f1': 1.0, 'baseline_accuracy': .1}
    with pytest.raises(IntegrityError):
        save_run(engine, str(uuid4()), metrics, 'a' * 64, 'b' * 64, 'model.joblib',
                 [{'id': 999999}], [1], [.9])
    assert summary(engine)['row_counts'] == {'samples': 1797, 'training_runs': 0, 'predictions': 0}


def check_training_round_trip(engine, tmp_path):
    seed_database(engine)
    metrics = train_database(engine, tmp_path)
    result = summary(engine)
    assert result['row_counts'] == {'samples': 1797, 'training_runs': 1, 'predictions': 450}
    assert result['latest_run']['id'] == metrics['run_id']
    assert result['latest_run']['accuracy'] > .93
    assert result['latest_run']['model_sha256'] == metrics['model_sha256']
    assert sum(r['errors'] for r in result['latest_errors_by_digit']) == round((1 - metrics['accuracy']) * 450)
    assert (tmp_path / result['latest_run']['model_file']).is_file()
    with engine.connect() as conn:
        # No predictions are accidentally logged against training samples.
        query = select(func.count()).select_from(predictions.join(samples)).where(samples.c.split == 'train')
        assert conn.scalar(query) == 0
    saved = json.loads((tmp_path / 'runs' / metrics['run_id'] / 'metrics.json').read_text())
    assert saved['dataset_sha256'] == metrics['dataset_sha256']


def test_training_reads_from_sql_and_saves_a_complete_run(engine, tmp_path):
    check_training_round_trip(engine, tmp_path)


def test_missing_connection_configuration_is_clear(monkeypatch):
    monkeypatch.delenv('DATABASE_URL', raising=False)
    with pytest.raises(ValueError, match='DATABASE_URL'):
        connect()


def test_postgresql_round_trip(tmp_path):
    url = os.environ.get('TEST_DATABASE_URL')
    if not url:
        pytest.skip('Run the Compose test service for the PostgreSQL integration check.')
    base = connect(url)
    schema = 'test_' + uuid4().hex
    try:
        with base.begin() as conn:
            conn.execute(CreateSchema(schema))
        db = base.execution_options(schema_translate_map={None: schema})
        check_training_round_trip(db, tmp_path)
        assert not seed_database(db)
    finally:
        with base.begin() as conn:
            conn.execute(DropSchema(schema, cascade=True, if_exists=True))
        base.dispose()
