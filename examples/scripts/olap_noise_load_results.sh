export NOISE_OLAP_DB_URL="postgresql+psycopg2://metabase:metabase_pass@localhost:5432/noise_sensitivity_olap"

python examples/olap/load_noise_sensitivity_results.py ./results/noise_sensitivity
