@echo off
setlocal enabledelayedexpansion

:: Define the configuration directory
set "CONFIG_DIR=examples\config\phase_1"

:: Iterate over all JSON files in the directory
for %%F in ("%CONFIG_DIR%\*.json") do (
    echo Running preprocessor with configuration: %%~nxF
    predictor.bat --load_config "%%F"
)

echo All configurations processed.
endlocal
pause
