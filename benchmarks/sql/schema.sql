CREATE TABLE IF NOT EXISTS bench_run (
  run_sha       text NOT NULL,
  parser        text NOT NULL,
  ci_run_id     bigint,
  created_at    timestamptz DEFAULT now(),
  mode          text,
  n_records     int,
  n_errors      int,
  elapsed_s     real,
  dataset                 jsonb,
  llm_config              jsonb,
  tokens_total            jsonb,
  tokens_by_stage         jsonb,
  tokens_by_metric_group  jsonb,
  PRIMARY KEY (run_sha, parser)
);

CREATE TABLE IF NOT EXISTS bench_metric (
  run_sha   text NOT NULL,
  parser    text NOT NULL,
  corpus    text NOT NULL,
  metric    text NOT NULL,
  system    text NOT NULL,
  mean      double precision,
  ci_low    double precision,
  ci_high   double precision,
  delta_llm_minus_parser    double precision,
  wilcoxon_p_llm_vs_parser  double precision,
  vs_baseline               jsonb,
  PRIMARY KEY (run_sha, parser, corpus, metric, system),
  FOREIGN KEY (run_sha, parser) REFERENCES bench_run(run_sha, parser) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS bench_document (
  run_sha   text NOT NULL,
  parser    text NOT NULL,
  record_id text NOT NULL,
  corpus    text NOT NULL,
  parser_metrics jsonb,
  llm_metrics    jsonb,
  parser_pred    jsonb,
  llm_pred       jsonb,
  gold           jsonb,
  tokens         jsonb,
  PRIMARY KEY (run_sha, parser, record_id),
  FOREIGN KEY (run_sha, parser) REFERENCES bench_run(run_sha, parser) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS bench_metric_corpus_metric_idx
  ON bench_metric (corpus, metric, system);
CREATE INDEX IF NOT EXISTS bench_run_created_idx
  ON bench_run (created_at DESC);
