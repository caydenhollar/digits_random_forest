"""Small SQLAlchemy data layer; PostgreSQL is the demo database."""

from datetime import datetime, timezone
import hashlib
import json
import os

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sqlalchemy import (CheckConstraint, Column, Float, ForeignKey, Integer, MetaData,
                        String, Table, Text, create_engine, event, func, select)

metadata = MetaData()
samples = Table('samples', metadata,
    Column('id', Integer, primary_key=True, autoincrement=False),
    Column('label', Integer, nullable=False), Column('split', String(5), nullable=False),
    Column('split_order', Integer, nullable=False), Column('pixels_json', Text, nullable=False),
    CheckConstraint('label BETWEEN 0 AND 9'), CheckConstraint("split IN ('train', 'test')"))
runs = Table('training_runs', metadata,
    Column('id', String(36), primary_key=True), Column('created_at', String(40), nullable=False),
    Column('seed', Integer, nullable=False), Column('dataset_sha256', String(64), nullable=False),
    Column('model_sha256', String(64), nullable=False), Column('model_file', Text, nullable=False),
    Column('accuracy', Float, nullable=False), Column('macro_f1', Float, nullable=False),
    Column('baseline_accuracy', Float, nullable=False))
predictions = Table('predictions', metadata,
    Column('run_id', String(36), ForeignKey('training_runs.id'), primary_key=True),
    Column('sample_id', Integer, ForeignKey('samples.id'), primary_key=True),
    Column('predicted_label', Integer, nullable=False), Column('vote_share', Float, nullable=False),
    CheckConstraint('predicted_label BETWEEN 0 AND 9'),
    CheckConstraint('vote_share >= 0 AND vote_share <= 1'))


def connect(url=None):
    url = url or os.environ.get('DATABASE_URL')
    if not url:
        raise ValueError('Set DATABASE_URL, or use Docker Compose for the PostgreSQL demo.')
    engine = create_engine(url, pool_pre_ping=True)
    if engine.dialect.name == 'sqlite':
        event.listen(engine, 'connect', lambda conn, _: conn.execute('PRAGMA foreign_keys=ON'))
    return engine


def seed_database(engine, seed=42):
    """Idempotent import: preserve existing data and reject a different split."""
    metadata.create_all(engine)
    data = load_digits()
    train_ids, test_ids = train_test_split(np.arange(len(data.target)), test_size=.25,
                                          random_state=seed, stratify=data.target)
    rows = [{'id': int(i), 'label': int(data.target[i]), 'split': split,
             'split_order': order, 'pixels_json': json.dumps(data.data[i].astype(int).tolist())}
            for split, ids in [('train', train_ids), ('test', test_ids)]
            for order, i in enumerate(ids)]
    with engine.begin() as conn:
        existing = [dict(r) for r in conn.execute(select(samples).order_by(samples.c.id)).mappings()]
        if existing:
            if existing != sorted(rows, key=lambda r: r['id']):
                raise ValueError('Database samples differ. Use a new database for a different dataset or split.')
            return False
        conn.execute(samples.insert(), rows)
    return True


def load_data(engine):
    groups = {}
    with engine.connect() as conn:
        for split in ('train', 'test'):
            groups[split] = [dict(r) for r in conn.execute(
                select(samples).where(samples.c.split == split).order_by(samples.c.split_order)).mappings()]
    if not groups['train'] or not groups['test']:
        raise ValueError('Both train and test samples are required; run db-seed first.')
    arrays = {}
    for split, rows in groups.items():
        try:
            pixels = np.asarray([json.loads(r['pixels_json']) for r in rows], dtype=float)
        except (TypeError, ValueError):
            raise ValueError('Database samples must contain 64 finite pixel values between 0 and 16.') from None
        if (pixels.shape != (len(rows), 64) or not np.isfinite(pixels).all()
                or (pixels < 0).any() or (pixels > 16).any()):
            raise ValueError('Database samples must contain 64 finite pixel values between 0 and 16.')
        arrays[split] = (pixels, np.asarray([r['label'] for r in rows]))
    rows = sorted(groups['train'] + groups['test'], key=lambda r: r['id'])
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    data = (arrays['train'][0], arrays['test'][0], arrays['train'][1], arrays['test'][1])
    return data, groups['test'], digest


def save_run(engine, run_id, metrics, dataset_hash, model_hash, model_file, test_rows, labels, votes):
    run = {'id': run_id, 'created_at': datetime.now(timezone.utc).isoformat(),
           'seed': metrics['seed'], 'dataset_sha256': dataset_hash,
           'model_sha256': model_hash, 'model_file': model_file,
           **{k: metrics[k] for k in ('accuracy', 'macro_f1', 'baseline_accuracy')}}
    rows = [{'run_id': run_id, 'sample_id': row['id'], 'predicted_label': int(label),
             'vote_share': float(vote)} for row, label, vote in zip(test_rows, labels, votes, strict=True)]
    # A failed prediction insert rolls back the run as well.
    with engine.begin() as conn:
        conn.execute(runs.insert(), run)
        conn.execute(predictions.insert(), rows)


def summary(engine):
    with engine.connect() as conn:
        counts = {table.name: conn.scalar(select(func.count()).select_from(table))
                  for table in (samples, runs, predictions)}
        latest = conn.execute(select(runs).order_by(runs.c.created_at.desc(), runs.c.id).limit(1)).mappings().first()
        errors = []
        if latest:
            query = (select(samples.c.label, func.count().label('errors'))
                     .select_from(predictions.join(samples))
                     .where(predictions.c.run_id == latest['id'],
                            predictions.c.predicted_label != samples.c.label)
                     .group_by(samples.c.label).order_by(samples.c.label))
            errors = [dict(r) for r in conn.execute(query).mappings()]
    return {'row_counts': counts, 'latest_run': dict(latest) if latest else None,
            'latest_errors_by_digit': errors}
