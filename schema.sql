CREATE TABLE IF NOT EXISTS fact_observation (
    metric_id   text             NOT NULL,
    obs_date    date             NOT NULL,
    raw_value   double precision,
    source      text             NOT NULL,
    series_id   text,
    valid_as_of timestamptz      NOT NULL,
    PRIMARY KEY (metric_id, obs_date)
);
