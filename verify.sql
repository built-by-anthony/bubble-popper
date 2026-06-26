-- Row count per series (daily should dwarf monthly)
SELECT metric_id, series_id, COUNT(*) AS row_count
FROM fact_observation
GROUP BY metric_id, series_id
ORDER BY row_count DESC;

-- Date range per series
SELECT metric_id, series_id, MIN(obs_date) AS first_obs, MAX(obs_date) AS last_obs
FROM fact_observation
GROUP BY metric_id, series_id
ORDER BY metric_id;

-- Spot-check: most recent 5 fed_funds values
SELECT obs_date, raw_value, valid_as_of
FROM fact_observation
WHERE metric_id = 'fed_funds'
ORDER BY obs_date DESC
LIMIT 5;

-- Any nulls snuck through?
SELECT metric_id, COUNT(*) AS null_values
FROM fact_observation
WHERE raw_value IS NULL
GROUP BY metric_id;
