:: Define the configuration directory
set "CONFIG_DIR=examples\config\phase_1"

:: Iterate over all JSON files in the directory

call predictor.bat --load_config %CONFIG_DIR%\phase_1_base_1575_1h_config.json

call predictor.bat --load_config %CONFIG_DIR%\phase_1_base_3150_1h_config.json

call predictor.bat --load_config %CONFIG_DIR%\phase_1_base_6300_1h_config.json

call predictor.bat --load_config %CONFIG_DIR%\phase_1_base_12600_1h_config.json

call predictor.bat --load_config %CONFIG_DIR%\phase_1_base_25200_1h_config.json



echo All configurations processed.

