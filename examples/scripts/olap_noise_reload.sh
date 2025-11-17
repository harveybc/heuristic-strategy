export NOISE_OLAP_DB_URL="postgresql+psycopg2://metabase:metabase_pass@localhost:5432/noise_sensitivity_olap"
python olap/reset_and_reload_noise_sensitivity_olap.py ./examples/results/noise_sensitivity
