-- Confirm the imported data split and class balance.
SELECT split, label, COUNT(*) AS samples
FROM samples GROUP BY split, label ORDER BY split, label;

-- Compare recent model runs, including the dataset and model identities.
SELECT id, created_at, seed, accuracy, macro_f1, baseline_accuracy,
       dataset_sha256, model_sha256
FROM training_runs ORDER BY created_at DESC LIMIT 5;

-- Join predictions to their actual labels and group errors by run and digit.
SELECT p.run_id, s.label AS actual_digit, COUNT(*) AS mistakes
FROM predictions p JOIN samples s ON s.id = p.sample_id
WHERE p.predicted_label <> s.label
GROUP BY p.run_id, s.label ORDER BY p.run_id, s.label;
