import pandas as pd
import numpy as np
import time
from app.data_handler import load_csv
from app.optimizer import run_optimizer

# =============================================================================
# DATA PROCESSOR FOR TRADING STRATEGY OPTIMIZATION
# =============================================================================
#
# Processes datasets and runs the optimizer.
# Calls `run_optimizer()` from `app.optimizer` to optimize the trading strategy.
# =============================================================================

def create_hourly_predictions(df, horizon):
    """
    Auto-compute hourly predictions from the base dataset.
    """
    blocks = []
    for i in range(len(df) - horizon):
        block = df.iloc[i+1 : i+1+horizon].values.flatten()
        blocks.append(block)
    return pd.DataFrame(blocks, index=df.index[:-horizon])

def create_daily_predictions(df, horizon):
    """
    Auto-compute daily predictions from the base dataset.

    Each row i in 'df' yields a block of predicted values at offsets t+24, t+48, ... t+24*horizon.
    If there are not enough rows (or any row indexing is invalid), we return an empty DataFrame.
    """
    nrows = len(df)
    required_rows = horizon * 24

    # If the dataset is too short for daily predictions, return empty.
    if nrows < required_rows:
        print("Warning: Not enough rows to create daily predictions. Returning an empty DataFrame.")
        return pd.DataFrame()

    blocks = []
    # For each row i, gather the next horizon days (24*horizon ticks).
    for i in range(nrows - required_rows):
        block_values = []
        # Attempt to fetch each day offset at i + d*24
        for d in range(1, horizon + 1):
            idx = i + d * 24
            # If idx is out of range for some reason (edge cases), skip safely
            if idx >= nrows:
                break
            row_vals = df.iloc[idx].values  # shape: (features,)
            if row_vals.ndim == 0:
                # means we have an empty row or zero-dimensional array
                continue
            block_values.extend(row_vals.flatten())
        # If we got some values, append them to blocks
        if block_values:
            blocks.append(block_values)
        else:
            # If we ended up with no valid data for that block, skip
            continue

    if not blocks:
        # No valid blocks were created
        print("Warning: daily predictions are empty after alignment. Returning an empty DataFrame.")
        return pd.DataFrame()

    # The resulting blocks length = nrows - required_rows
    # Align the index accordingly
    daily_idx = df.index[:(nrows - required_rows)]
    return pd.DataFrame(blocks, index=daily_idx)



def process_data(config):
    """
    Loads and processes datasets, ensuring alignment and applying max_steps.
    - Uses external prediction files if provided.
    - Generates predictions if files are not available.
    - Ensures all datasets (base, hourly predictions, daily predictions) are properly aligned to the common date range.
      Additionally, preserves the full base dataset (base_full) for later evaluation.
    """
    import pandas as pd
    import numpy as np
    import json
    from app.data_handler import load_csv

    headers = config.get("headers", True)
    print("Loading datasets...")

    # Load datasets based on config parameters
    hourly_df = load_csv(config["hourly_predictions_file"], headers=headers) if config.get("hourly_predictions_file") else None
    daily_df = load_csv(config["daily_predictions_file"], headers=headers) if config.get("daily_predictions_file") else None
    base_df = load_csv(config["base_dataset_file"], headers=headers)
    print(f"Base dataset loaded: {base_df.shape}")

    # Conserva una copia completa del base dataset para la evaluación
    base_df_full = base_df.copy()

    # Auto-generate predictions if files are missing
    if hourly_df is None:
        if "time_horizon" not in config or not config["time_horizon"]:
            raise ValueError("time_horizon must be provided when auto-generating predictions.")
        print("Auto-generating hourly predictions...")
        hourly_df = create_hourly_predictions(base_df, config["time_horizon"])

    if daily_df is None:
        if "time_horizon" not in config or not config["time_horizon"]:
            raise ValueError("time_horizon must be provided when auto-generating predictions.")
        print("Auto-generating daily predictions...")
        daily_df = create_daily_predictions(base_df, config["time_horizon"])

    # Ensure that hourly and daily predictions have a datetime index based on DATE_TIME column, if not already set.
    if hourly_df is not None:
        if not isinstance(hourly_df.index, pd.DatetimeIndex):
            if "DATE_TIME" in hourly_df.columns:
                hourly_df.index = pd.to_datetime(hourly_df["DATE_TIME"])
            else:
                raise ValueError("hourly_df does not have a DATE_TIME column.")
    if daily_df is not None:
        if not isinstance(daily_df.index, pd.DatetimeIndex):
            if "DATE_TIME" in daily_df.columns:
                daily_df.index = pd.to_datetime(daily_df["DATE_TIME"])
            else:
                raise ValueError("daily_df does not have a DATE_TIME column.")

    # Ensure base_df has a datetime index as well.
    if not isinstance(base_df.index, pd.DatetimeIndex):
        if "DATE_TIME" in base_df.columns:
            base_df.index = pd.to_datetime(base_df["DATE_TIME"])
        else:
            raise ValueError("base_df does not have a DATE_TIME column.")

    # Align all datasets to the common date range (intersection of their datetime indexes)
    common_index = base_df.index.intersection(hourly_df.index).intersection(daily_df.index)
    if common_index.empty:
        raise ValueError("No common date range found among base, hourly, and daily predictions.")

    # Aquí se recortan los tres datasets al rango común
    base_df = base_df.loc[common_index]         # <-- Alinea el dataset base
    hourly_df = hourly_df.loc[common_index]
    daily_df = daily_df.loc[common_index]

    # Verify that all datasets have the same number of rows
    if not (len(base_df) == len(hourly_df) == len(daily_df)):
        raise ValueError("After alignment, the number of rows in base, hourly, and daily predictions do not match!")

    # Print aligned date ranges
    print(f"Aligned Base dataset range: {base_df.index.min()} to {base_df.index.max()}")
    print(f"Aligned Hourly predictions range: {hourly_df.index.min()} to {hourly_df.index.max()}")
    print(f"Aligned Daily predictions range: {daily_df.index.min()} to {daily_df.index.max()}")

    return {"hourly": hourly_df, "daily": daily_df, "base": base_df, "base_full": base_df_full}


7 días anteriores
30 días anteriores
enero
2024
Dijiste:
por favor revisa si al procesar los datos este script alinea las fechas de los datasets horario y diario al ser cargados como archivos, dime en que linea exactamente se hace la alineación:
import pandas as pd
import numpy as np
import time
from app.data_handler import load_csv
from app.optimizer import run_optimizer

# =============================================================================
# DATA PROCESSOR FOR TRADING STRATEGY OPTIMIZATION
# =============================================================================
#
# Processes datasets and runs the optimizer.
# Calls run_optimizer() from app.optimizer to optimize the trading strategy.
# =============================================================================

def create_hourly_predictions(df, horizon):
    """
    Auto-compute hourly predictions from the base dataset.
    """
    blocks = []
    for i in range(len(df) - horizon):
        block = df.iloc[i+1 : i+1+horizon].values.flatten()
        blocks.append(block)
    return pd.DataFrame(blocks, index=df.index[:-horizon])

def create_daily_predictions(df, horizon):
    """
    Auto-compute daily predictions from the base dataset.

    Each row i in 'df' yields a block of predicted values at offsets t+24, t+48, ... t+24*horizon.
    If there are not enough rows (or any row indexing is invalid), we return an empty DataFrame.
    """
    nrows = len(df)
    required_rows = horizon * 24

    # If the dataset is too short for daily predictions, return empty.
    if nrows < required_rows:
        print("Warning: Not enough rows to create daily predictions. Returning an empty DataFrame.")
        return pd.DataFrame()

    blocks = []
    # For each row i, gather the next horizon days (24*horizon ticks).
    for i in range(nrows - required_rows):
        block_values = []
        # Attempt to fetch each day offset at i + d*24
        for d in range(1, horizon + 1):
            idx = i + d * 24
            # If idx is out of range for some reason (edge cases), skip safely
            if idx >= nrows:
                break
            row_vals = df.iloc[idx].values  # shape: (features,)
            if row_vals.ndim == 0:
                # means we have an empty row or zero-dimensional array
                continue
            block_values.extend(row_vals.flatten())
        # If we got some values, append them to blocks
        if block_values:
            blocks.append(block_values)
        else:
            # If we ended up with no valid data for that block, skip
            continue

    if not blocks:
        # No valid blocks were created
        print("Warning: daily predictions are empty after alignment. Returning an empty DataFrame.")
        return pd.DataFrame()

    # The resulting blocks length = nrows - required_rows
    # Align the index accordingly
    daily_idx = df.index[:(nrows - required_rows)]
    return pd.DataFrame(blocks, index=daily_idx)



def process_data(config):
    """
    Loads and processes datasets, ensuring alignment and applying max_steps.
    - Uses external prediction files if provided.
    - Generates predictions if files are not available.
    - Ensures all datasets are properly aligned.
    """
    import pandas as pd
    import numpy as np
    import json
    from app.data_handler import load_csv

    headers = config.get("headers", True)
    print("Loading datasets...")

    # Load datasets based on config parameters
    hourly_df = load_csv(config["hourly_predictions_file"], headers=headers) if config.get("hourly_predictions_file") else None
    daily_df = load_csv(config["daily_predictions_file"], headers=headers) if config.get("daily_predictions_file") else None
    base_df = load_csv(config["base_dataset_file"], headers=headers)

    print(f"Base dataset loaded: {base_df.shape}")

    # Auto-generate predictions if files are missing
    if hourly_df is None:
        if "time_horizon" not in config or not config["time_horizon"]:
            raise ValueError("time_horizon must be provided when auto-generating predictions.")
        print("Auto-generating hourly predictions...")
        hourly_df = create_hourly_predictions(base_df, config["time_horizon"])

    if daily_df is None:
        if "time_horizon" not in config or not config["time_horizon"]:
            raise ValueError("time_horizon must be provided when auto-generating predictions.")
        print("Auto-generating daily predictions...")
        daily_df = create_daily_predictions(base_df, config["time_horizon"])

    # Align datasets if they have datetime indexes
    if isinstance(base_df.index, pd.DatetimeIndex):
        for label, df in zip(["Hourly", "Daily"], [hourly_df, daily_df]):
            if df is not None and not isinstance(df.index, pd.DatetimeIndex):
                print(f"Warning: {label} dataset does not have a datetime index. Assigning base dataset index.")
                df.index = base_df.index[:len(df)]
                df.index.name = "DATE_TIME"

    return {"hourly": hourly_df, "daily": daily_df, "base": base_df}

def run_processing_pipeline(config, plugin):
    """
    Executes the trading strategy optimization pipeline.
    
    - Loads and processes datasets.
    - If config["load_parameters"] is provided (not None), loads candidate parameters from the specified JSON file
      and evaluates the strategy once using those parameters (printing all trades, summary, and saving outputs).
    - Otherwise, runs the full optimization via run_optimizer().
    - At the end, renames the balance plot and saves the trades and summary CSV files.
    - If in optimization mode and if config["save_parameters"] is provided, saves the best parameters as JSON.
    """
    import json, os, pandas as pd
    from app.optimizer import init_optimizer, evaluate_individual, run_optimizer
    import time

    start_time = time.time()
    strat_name = config.get("strategy_name", "Heuristic Strategy")
    print(f"\n=== Starting Trading Strategy Optimization Pipeline for '{strat_name}' ===")

    datasets = process_data(config)
    hourly_preds, daily_preds, base_data = datasets["hourly"], datasets["daily"], datasets["base"]

    print("\nProcessed Dataset Shapes:")
    print(f"  Hourly predictions: {hourly_preds.shape}")
    print(f"  Daily predictions:  {daily_preds.shape}")
    print(f"  Base dataset:       {base_data.shape}")

    # If load_parameters is provided, load candidate parameters and evaluate the strategy once.
    if config.get("load_parameters") is not None:
        try:
            with open(config["load_parameters"], "r") as f:
                loaded_params = json.load(f)
            print(f"Loaded evaluation parameters from {config['load_parameters']}: {loaded_params}")
        except Exception as e:
            print(f"Failed to load parameters from {config['load_parameters']}: {e}")
            loaded_params = None
        if loaded_params is not None:
            # Construct candidate in the same order as get_optimizable_params():
            candidate = [
                loaded_params.get("profit_threshold", plugin.params["profit_threshold"]),
                loaded_params.get("tp_multiplier", plugin.params["tp_multiplier"]),
                loaded_params.get("sl_multiplier", plugin.params["sl_multiplier"]),
                loaded_params.get("lower_rr_threshold", plugin.params["lower_rr_threshold"]),
                loaded_params.get("upper_rr_threshold", plugin.params["upper_rr_threshold"]),
                int(loaded_params.get("time_horizon", 3))
            ]
            print(f"Evaluating strategy with loaded parameters: {candidate}")
            # Initialize optimizer globals so that evaluate_individual() can use them
            init_optimizer(plugin, base_data, hourly_preds, daily_preds, config)
            result = evaluate_individual(candidate)
            trading_info = {"best_parameters": {
                "profit_threshold": candidate[0],
                "tp_multiplier": candidate[1],
                "sl_multiplier": candidate[2],
                "lower_rr_threshold": candidate[3],
                "upper_rr_threshold": candidate[4],
                "time_horizon": candidate[5]
            }, "profit": result[0]}
        else:
            trading_info = {}
    else:
        # Otherwise, run full optimization.
        if hasattr(plugin, "get_optimizable_params") and hasattr(plugin, "evaluate_candidate"):
            print(f"\nPlugin supports optimization. Running optimizer for '{strat_name}'...")
            trading_info = run_optimizer(plugin, base_data, hourly_preds, daily_preds, config)
        else:
            print("\nPlugin does not support optimization. Exiting.")
            trading_info = {}

    print("\n=== Optimization Results ===")
    for key, value in trading_info.items():
        print(f"{key}: {value}")

    # Rename the final balance plot if 'balance_plot_file' is provided.
    if config.get("balance_plot_file"):
        old_plot = "balance_plot.png"
        new_plot = config["balance_plot_file"]
        if os.path.exists(old_plot):
            try:
                os.rename(old_plot, new_plot)
                print(f"Renamed {old_plot} -> {new_plot}")
            except Exception as e:
                print(f"Failed to rename {old_plot} to {new_plot}: {e}")
        else:
            print(f"Warning: {old_plot} not found; no balance plot to rename.")

    # Save trades to CSV if 'trades_csv_file' is specified.
    trades_csv = config.get("trades_csv_file")
    if trades_csv:
        try:
            if hasattr(plugin, "trades") and plugin.trades:
                pd.DataFrame(plugin.trades).to_csv(trades_csv, index=False)
                print(f"Trades saved to {trades_csv}.")
            else:
                print("Warning: plugin.trades not found or empty.")
        except Exception as e:
            print(f"Failed to save trades to {trades_csv}: {e}")

    # Save summary CSV if 'summary_csv_file' is specified.
    summary_csv = config.get("summary_csv_file")
    if summary_csv:
        try:
            df = pd.DataFrame([trading_info])
            df.to_csv(summary_csv, index=False)
            print(f"Summary saved to {summary_csv}.")
        except Exception as e:
            print(f"Failed to save summary CSV to {summary_csv}: {e}")

    # Save best parameters as JSON if in optimization mode and load_parameters is not provided.
    if config.get("load_parameters") is None and config.get("save_parameters"):
        try:
            with open(config["save_parameters"], "w") as f:
                json.dump(trading_info.get("best_parameters", {}), f, indent=4, default=str)
            print(f"Best parameters saved to {config['save_parameters']}.")
        except Exception as e:
            print(f"Failed to save best parameters to {config['save_parameters']}: {e}")

    end_time = time.time()
    print(f"\nTotal Execution Time: {end_time - start_time:.2f} seconds")
    return trading_info, getattr(plugin, "trades", None)



if __name__ == "__main__":
    pass
Razonado durante 32 segundos
La alineación se realiza en la función process_data. En concreto, si el índice del dataset base es de tipo datetime, se asigna a los datasets horario y diario (si no tienen un índice datetime) una porción del índice del dataset base. Esto se lleva a cabo en la línea 113, donde se ejecuta:

df.index = base_df.index[:len(df)]




Dijiste:
no me entiendes amigo, por favor enfócate y comprende esto, los archivos cargados desde hourly_predictions_file y daily_predictions_file son CSV, que ambos tienen la columna DATE_TIME pero no están alineados, es decir, als fechas de inicio y final de cada uno pueden ser diferentes (e hecho lo son con los que uso ahora para probar), entonces dime se hace alineacion o no y donde? porque se supone que nuestra estrategia heuristica (el plugin cargado) usa los datos y estos ya deben estar alineados, es decir los datos que se pasan  para realizar la estrategia tanto de las predicciones diarias, como las horarias (ambas tienen periodicidad de una hora, la predicción diaria contiene las predicciones para las siguientes 1 a 6 horas a partir del DATE_TIME, mientras que las predicciones diarias son para  el siguiente dia(24 horas) a 6 dias(6*24 horas) después de la fecha DATE_TIME ) entonces los datos pasados al plugin para la estrategia deben tener el mismo numero de filas, y deben ser las correspondientes al mismo DATE_TIME (RECUERDA, no debes por ningun motivo sobreescribir ninguna fecha, sino alinear los datasets y trimear las filas tanto del inicio como del final que no estén dentro del rango común de fechas) de forma que el dataset reciba para un mismo dia y hora, tanto las predicciones horarias  como las diarias,. 

Entonces, se está haciendo esto o no y exáctamente en que líneas de código?
Razonado durante 40 segundos
No, el script no realiza la alineación de los datasets de forma adecuada según un rango común de fechas. Lo que hace es, en la función process_data, verificar si el índice del dataset base es un DatetimeIndex y, en caso de que los datasets de predicción (horario y diario) no lo sean, les asigna un índice tomado de las primeras filas del dataset base. Esto se hace en el siguiente bloque de código:

# Align datasets if they have datetime indexes
if isinstance(base_df.index, pd.DatetimeIndex):
    for label, df in zip(["Hourly", "Daily"], [hourly_df, daily_df]):
        if df is not None and not isinstance(df.index, pd.DatetimeIndex):
            print(f"Warning: {label} dataset does not have a datetime index. Assigning base dataset index.")
            df.index = base_df.index[:len(df)]
            df.index.name = "DATE_TIME"
(aproximadamente líneas 105–110 en este script)

Esta asignación simplemente sobreescribe el índice de los archivos CSV cargados con una porción del índice del dataset base, sin recortar ni ajustar las filas para que ambas predicciones (horaria y diaria) correspondan exactamente al mismo rango de fechas (es decir, sin eliminar las filas que estén fuera del rango común). Por ello, si las fechas de inicio y final de los archivos CSV son diferentes, esta “alineación” no garantiza que los datasets tengan las mismas filas ni que se correspondan correctamente por DATE_TIME para la estrategia.





Dijiste:
está mal, corrigelo, tya sabes como debe funcionar, se deben recortar ambos daasets al rango de fechas y horas en común (DATETIME), además de hacerlo, antes de pasar estos datos al plugin, porfa realiza una verfificación adicional e imprime las fechas (y hora) de inicio y final de las predicciones diarias y horarias que se pasan finalmente al plugin. Por favor realiza las correcciones correspondientes y muéstrame el código completo, repito, el código completo para los métodos que requieren actualización únicamente, este es el último código del data_processor.py:
import pandas as pd
import numpy as np
import time
from app.data_handler import load_csv
from app.optimizer import run_optimizer

# =============================================================================
# DATA PROCESSOR FOR TRADING STRATEGY OPTIMIZATION
# =============================================================================
#
# Processes datasets and runs the optimizer.
# Calls run_optimizer() from app.optimizer to optimize the trading strategy.
# =============================================================================

def create_hourly_predictions(df, horizon):
    """
    Auto-compute hourly predictions from the base dataset.
    """
    blocks = []
    for i in range(len(df) - horizon):
        block = df.iloc[i+1 : i+1+horizon].values.flatten()
        blocks.append(block)
    return pd.DataFrame(blocks, index=df.index[:-horizon])

def create_daily_predictions(df, horizon):
    """
    Auto-compute daily predictions from the base dataset.

    Each row i in 'df' yields a block of predicted values at offsets t+24, t+48, ... t+24*horizon.
    If there are not enough rows (or any row indexing is invalid), we return an empty DataFrame.
    """
    nrows = len(df)
    required_rows = horizon * 24

    # If the dataset is too short for daily predictions, return empty.
    if nrows < required_rows:
        print("Warning: Not enough rows to create daily predictions. Returning an empty DataFrame.")
        return pd.DataFrame()

    blocks = []
    # For each row i, gather the next horizon days (24*horizon ticks).
    for i in range(nrows - required_rows):
        block_values = []
        # Attempt to fetch each day offset at i + d*24
        for d in range(1, horizon + 1):
            idx = i + d * 24
            # If idx is out of range for some reason (edge cases), skip safely
            if idx >= nrows:
                break
            row_vals = df.iloc[idx].values  # shape: (features,)
            if row_vals.ndim == 0:
                # means we have an empty row or zero-dimensional array
                continue
            block_values.extend(row_vals.flatten())
        # If we got some values, append them to blocks
        if block_values:
            blocks.append(block_values)
        else:
            # If we ended up with no valid data for that block, skip
            continue

    if not blocks:
        # No valid blocks were created
        print("Warning: daily predictions are empty after alignment. Returning an empty DataFrame.")
        return pd.DataFrame()

    # The resulting blocks length = nrows - required_rows
    # Align the index accordingly
    daily_idx = df.index[:(nrows - required_rows)]
    return pd.DataFrame(blocks, index=daily_idx)



def process_data(config):
    """
    Loads and processes datasets, ensuring alignment and applying max_steps.
    - Uses external prediction files if provided.
    - Generates predictions if files are not available.
    - Ensures all datasets are properly aligned.
    """
    import pandas as pd
    import numpy as np
    import json
    from app.data_handler import load_csv

    headers = config.get("headers", True)
    print("Loading datasets...")

    # Load datasets based on config parameters
    hourly_df = load_csv(config["hourly_predictions_file"], headers=headers) if config.get("hourly_predictions_file") else None
    daily_df = load_csv(config["daily_predictions_file"], headers=headers) if config.get("daily_predictions_file") else None
    base_df = load_csv(config["base_dataset_file"], headers=headers)

    print(f"Base dataset loaded: {base_df.shape}")

    # Auto-generate predictions if files are missing
    if hourly_df is None:
        if "time_horizon" not in config or not config["time_horizon"]:
            raise ValueError("time_horizon must be provided when auto-generating predictions.")
        print("Auto-generating hourly predictions...")
        hourly_df = create_hourly_predictions(base_df, config["time_horizon"])

    if daily_df is None:
        if "time_horizon" not in config or not config["time_horizon"]:
            raise ValueError("time_horizon must be provided when auto-generating predictions.")
        print("Auto-generating daily predictions...")
        daily_df = create_daily_predictions(base_df, config["time_horizon"])

    # Align datasets if they have datetime indexes
    if isinstance(base_df.index, pd.DatetimeIndex):
        for label, df in zip(["Hourly", "Daily"], [hourly_df, daily_df]):
            if df is not None and not isinstance(df.index, pd.DatetimeIndex):
                print(f"Warning: {label} dataset does not have a datetime index. Assigning base dataset index.")
                df.index = base_df.index[:len(df)]
                df.index.name = "DATE_TIME"

    return {"hourly": hourly_df, "daily": daily_df, "base": base_df}

def run_processing_pipeline(config, plugin):
    """
    Executes the trading strategy optimization pipeline.
    
    - Loads and processes datasets.
    - If config["load_parameters"] is provided (not None), loads candidate parameters from the specified JSON file
      and evaluates the strategy once using those parameters (printing all trades, summary, and saving outputs).
    - Otherwise, runs the full optimization via run_optimizer().
    - At the end, renames the balance plot and saves the trades and summary CSV files.
    - If in optimization mode and if config["save_parameters"] is provided, saves the best parameters as JSON.
    """
    import json, os, pandas as pd
    from app.optimizer import init_optimizer, evaluate_individual, run_optimizer
    import time

    start_time = time.time()
    strat_name = config.get("strategy_name", "Heuristic Strategy")
    print(f"\n=== Starting Trading Strategy Optimization Pipeline for '{strat_name}' ===")

    datasets = process_data(config)
    hourly_preds, daily_preds, base_data = datasets["hourly"], datasets["daily"], datasets["base"]

    print("\nProcessed Dataset Shapes:")
    print(f"  Hourly predictions: {hourly_preds.shape}")
    print(f"  Daily predictions:  {daily_preds.shape}")
    print(f"  Base dataset:       {base_data.shape}")

    # If load_parameters is provided, load candidate parameters and evaluate the strategy once.
    if config.get("load_parameters") is not None:
        try:
            with open(config["load_parameters"], "r") as f:
                loaded_params = json.load(f)
            print(f"Loaded evaluation parameters from {config['load_parameters']}: {loaded_params}")
        except Exception as e:
            print(f"Failed to load parameters from {config['load_parameters']}: {e}")
            loaded_params = None
        if loaded_params is not None:
            # Construct candidate in the same order as get_optimizable_params():
            candidate = [
                loaded_params.get("profit_threshold", plugin.params["profit_threshold"]),
                loaded_params.get("tp_multiplier", plugin.params["tp_multiplier"]),
                loaded_params.get("sl_multiplier", plugin.params["sl_multiplier"]),
                loaded_params.get("lower_rr_threshold", plugin.params["lower_rr_threshold"]),
                loaded_params.get("upper_rr_threshold", plugin.params["upper_rr_threshold"]),
                int(loaded_params.get("time_horizon", 3))
            ]
            print(f"Evaluating strategy with loaded parameters: {candidate}")
            # Initialize optimizer globals so that evaluate_individual() can use them
            init_optimizer(plugin, base_data, hourly_preds, daily_preds, config)
            result = evaluate_individual(candidate)
            trading_info = {"best_parameters": {
                "profit_threshold": candidate[0],
                "tp_multiplier": candidate[1],
                "sl_multiplier": candidate[2],
                "lower_rr_threshold": candidate[3],
                "upper_rr_threshold": candidate[4],
                "time_horizon": candidate[5]
            }, "profit": result[0]}
        else:
            trading_info = {}
    else:
        # Otherwise, run full optimization.
        if hasattr(plugin, "get_optimizable_params") and hasattr(plugin, "evaluate_candidate"):
            print(f"\nPlugin supports optimization. Running optimizer for '{strat_name}'...")
            trading_info = run_optimizer(plugin, base_data, hourly_preds, daily_preds, config)
        else:
            print("\nPlugin does not support optimization. Exiting.")
            trading_info = {}

    print("\n=== Optimization Results ===")
    for key, value in trading_info.items():
        print(f"{key}: {value}")

    # Rename the final balance plot if 'balance_plot_file' is provided.
    if config.get("balance_plot_file"):
        old_plot = "balance_plot.png"
        new_plot = config["balance_plot_file"]
        if os.path.exists(old_plot):
            try:
                os.rename(old_plot, new_plot)
                print(f"Renamed {old_plot} -> {new_plot}")
            except Exception as e:
                print(f"Failed to rename {old_plot} to {new_plot}: {e}")
        else:
            print(f"Warning: {old_plot} not found; no balance plot to rename.")

    # Save trades to CSV if 'trades_csv_file' is specified.
    trades_csv = config.get("trades_csv_file")
    if trades_csv:
        try:
            if hasattr(plugin, "trades") and plugin.trades:
                pd.DataFrame(plugin.trades).to_csv(trades_csv, index=False)
                print(f"Trades saved to {trades_csv}.")
            else:
                print("Warning: plugin.trades not found or empty.")
        except Exception as e:
            print(f"Failed to save trades to {trades_csv}: {e}")

    # Save summary CSV if 'summary_csv_file' is specified.
    summary_csv = config.get("summary_csv_file")
    if summary_csv:
        try:
            df = pd.DataFrame([trading_info])
            df.to_csv(summary_csv, index=False)
            print(f"Summary saved to {summary_csv}.")
        except Exception as e:
            print(f"Failed to save summary CSV to {summary_csv}: {e}")

    # Save best parameters as JSON if in optimization mode and load_parameters is not provided.
    if config.get("load_parameters") is None and config.get("save_parameters"):
        try:
            with open(config["save_parameters"], "w") as f:
                json.dump(trading_info.get("best_parameters", {}), f, indent=4, default=str)
            print(f"Best parameters saved to {config['save_parameters']}.")
        except Exception as e:
            print(f"Failed to save best parameters to {config['save_parameters']}: {e}")

    end_time = time.time()
    print(f"\nTotal Execution Time: {end_time - start_time:.2f} seconds")
    return trading_info, getattr(plugin, "trades", None)



if __name__ == "__main__":
    pass


2/2

```python def process_data(config): """ Loads and processes datasets, ensuring alignment and applying max_steps. - Uses external prediction files if provided. - Generates predictions if files are not available. - Ensures all datasets are properly aligned. """ import pandas as pd import numpy as np import json from app.data_handler import load_csv headers = config.get("headers", True) print("Loading datasets...") # Load datasets based on config parameters hourly_df = load_csv(config["hourly_predictions_file"], headers=headers) if config.get("hourly_predictions_file") else None daily_df = load_csv(config["daily_predictions_file"], headers=headers) if config.get("daily_predictions_file") else None base_df = load_csv(config["base_dataset_file"], headers=headers) print(f"Base dataset loaded: {base_df.shape}") # Auto-generate predictions if files are missing if hourly_df is None: if "time_horizon" not in config or not config["time_horizon"]: raise ValueError("time_horizon must be provided when auto-generating predictions.") print("Auto-generating hourly predictions...") hourly_df = create_hourly_predictions(base_df, config["time_horizon"]) if daily_df is None: if "time_horizon" not in config or not config["time_horizon"]: raise ValueError("time_horizon must be provided when auto-generating predictions.") print("Auto-generating daily predictions...") daily_df = create_daily_predictions(base_df, config["time_horizon"]) # Ensure that datasets align based on the common datetime range if isinstance(base_df.index, pd.DatetimeIndex): for label, df in zip(["Hourly", "Daily"], [hourly_df, daily_df]): if df is not None: # Align to the common datetime range common_start = max(base_df.index.min(), df.index.min()) common_end = min(base_df.index.max(), df.index.max()) df = df[(df.index >= common_start) & (df.index <= common_end)] print(f'{label} dataset aligned: {common_start} to {common_end}') # Reassign the aligned datetime index df.index = df.index df.index.name = "DATE_TIME" # Print the aligned date/time ranges before passing to plugin for further use print("Final aligned dataset ranges:") print(f"Hourly predictions: {hourly_df.index.min()} to {hourly_df.index.max()}") print(f"Daily predictions: {daily_df.index.min()} to {daily_df.index.max()}") return {"hourly": hourly_df, "daily": daily_df, "base": base_df} ``` durante 42 segundos
Parece que el usuario está pidiendo ajustar ambos conjuntos de datos al rango de fechas y horas en común, usando la columna DATETIME. Además, me piden hacer una verificación extra para imprimir las fechas de inicio y final de las predicciones, tanto diarias como horarias, antes de pasarlas al plugin.

Voy a corregir eso, asegurándome de ajustar los datos correctamente y mostrar las fechas antes de pasarlas. Luego, compartiré únicamente el código actualizado de los métodos que lo requieren.

Necesito actualizar el código de la función process_data para alinear los datasets de predicciones horarias y diarias a un rango común de fechas, sin sobrescribir la columna DATE_TIME. Primero, debería usar la columna DATE_TIME para alinearlos, pero sin modificarla. Para esto, puedo filtrar ambos datasets para el rango de fechas en común, sin sobrescribir la columna de fechas. Después, imprimiré las fechas de inicio y fin de las predicciones antes de pasarlas al plugin.

Necesito asegurarnos de que los conjuntos de datos hourly_df y daily_df estén alineados según el rango común de fechas. Primero, verificamos si tienen un índice DatetimeIndex, y si no, asignamos la columna DATE_TIME como índice. Luego, calcularemos la intersección de estos índices para filtrar los datos según ese rango común y finalmente imprimimos las fechas de inicio y final de las predicciones. Realizaré esta alineación y la impresión en la función process_data antes de pasar los datos al plugin, sin modificar las fechas originales.

En la función process_data, voy a asegurarme de alinear los datasets de predicciones horarias y diarias según el rango común de fechas. Esto implica convertir los índices a DatetimeIndex si es necesario, utilizar la intersección para filtrar los datos y finalmente imprimir los rangos de fechas alineados. También agregaré una verificación para asegurarme de que ambas predicciones tengan el mismo número de filas después de la alineación. No es necesario modificar run_processing_pipeline, solo agregaré una verificación en process_data.

Voy a agregar una verificación en process_data para asegurarme de que las predicciones horarias y diarias tengan el mismo número de filas después de alinearlas. Luego, en run_processing_pipeline, agregaré impresión adicional para verificar los rangos de fechas de las predicciones justo antes de pasar los datos al plugin. Esto es redundante, pero cumple con la instrucción de imprimir las fechas de inicio y fin. Presentaré el código completo para ambas funciones, tal como se requiere.

A continuación te muestro los métodos actualizados. Se ha modificado el método process_data para:

• Convertir (si es necesario) los índices de las predicciones horarias y diarias a un DatetimeIndex usando la columna "DATE_TIME".
• Calcular la intersección de ambos índices (rango de fechas en común) y filtrar cada DataFrame para quedarse solo con ese rango.
• Imprimir las fechas (y horas) de inicio y final de los datos alineados.

Además, en run_processing_pipeline se imprime nuevamente el rango final de fechas que se pasa al plugin antes de la optimización.

Aquí tienes el código completo de los métodos actualizados:

def process_data(config):
    """
    Loads and processes datasets, ensuring alignment and applying max_steps.
    - Uses external prediction files if provided.
    - Generates predictions if files are not available.
    - Ensures all datasets are properly aligned to the common date range.
    """
    import pandas as pd
    import numpy as np
    import json
    from app.data_handler import load_csv

    headers = config.get("headers", True)
    print("Loading datasets...")

    # Load datasets based on config parameters
    hourly_df = load_csv(config["hourly_predictions_file"], headers=headers) if config.get("hourly_predictions_file") else None
    daily_df = load_csv(config["daily_predictions_file"], headers=headers) if config.get("daily_predictions_file") else None
    base_df = load_csv(config["base_dataset_file"], headers=headers)

    print(f"Base dataset loaded: {base_df.shape}")

    # Auto-generate predictions if files are missing
    if hourly_df is None:
        if "time_horizon" not in config or not config["time_horizon"]:
            raise ValueError("time_horizon must be provided when auto-generating predictions.")
        print("Auto-generating hourly predictions...")
        hourly_df = create_hourly_predictions(base_df, config["time_horizon"])

    if daily_df is None:
        if "time_horizon" not in config or not config["time_horizon"]:
            raise ValueError("time_horizon must be provided when auto-generating predictions.")
        print("Auto-generating daily predictions...")
        daily_df = create_daily_predictions(base_df, config["time_horizon"])

    # Ensure that hourly and daily predictions have a datetime index based on DATE_TIME column, if not already set.
    if hourly_df is not None:
        if not isinstance(hourly_df.index, pd.DatetimeIndex):
            if "DATE_TIME" in hourly_df.columns:
                hourly_df.index = pd.to_datetime(hourly_df["DATE_TIME"])
            else:
                raise ValueError("hourly_df does not have a DATE_TIME column.")
    if daily_df is not None:
        if not isinstance(daily_df.index, pd.DatetimeIndex):
            if "DATE_TIME" in daily_df.columns:
                daily_df.index = pd.to_datetime(daily_df["DATE_TIME"])
            else:
                raise ValueError("daily_df does not have a DATE_TIME column.")

    # Align datasets to the common date range (intersection of their datetime indexes)
    common_index = hourly_df.index.intersection(daily_df.index)
    if common_index.empty:
        raise ValueError("No common date range found between hourly and daily predictions.")
    hourly_df = hourly_df.loc[common_index]
    daily_df = daily_df.loc[common_index]

    # Verify that both datasets have the same number of rows
    if len(hourly_df) != len(daily_df):
        raise ValueError("After alignment, hourly and daily predictions have different number of rows!")

    # Print aligned date ranges
    print(f"Aligned Hourly predictions range: {hourly_df.index.min()} to {hourly_df.index.max()}")
    print(f"Aligned Daily predictions range: {daily_df.index.min()} to {daily_df.index.max()}")

    return {"hourly": hourly_df, "daily": daily_df, "base": base_df}


def run_processing_pipeline(config, plugin):
    """
    Executes the trading strategy optimization pipeline.
    
    - Loads and processes datasets.
    - If config["load_parameters"] is provided (not None), loads candidate parameters from the specified JSON file
      and evaluates the strategy once using those parameters (printing all trades, summary, and saving outputs).
    - Otherwise, runs the full optimization via run_optimizer().
    - At the end, renames the balance plot and saves the trades and summary CSV files.
    - If in optimization mode and if config["save_parameters"] is provided, saves the best parameters as JSON.
    """
    import json, os, pandas as pd
    from app.optimizer import init_optimizer, evaluate_individual, run_optimizer
    import time

    start_time = time.time()
    strat_name = config.get("strategy_name", "Heuristic Strategy")
    print(f"\n=== Starting Trading Strategy Optimization Pipeline for '{strat_name}' ===")

    datasets = process_data(config)
    hourly_preds, daily_preds, base_data = datasets["hourly"], datasets["daily"], datasets["base"]

    # Additional verification: print final aligned date ranges before passing data to the plugin
    print(f"Final Hourly predictions date range: {hourly_preds.index.min()} to {hourly_preds.index.max()}")
    print(f"Final Daily predictions date range: {daily_preds.index.min()} to {daily_preds.index.max()}")

    print("\nProcessed Dataset Shapes:")
    print(f"  Hourly predictions: {hourly_preds.shape}")
    print(f"  Daily predictions:  {daily_preds.shape}")
    print(f"  Base dataset:       {base_data.shape}")

    # If load_parameters is provided, load candidate parameters and evaluate the strategy once.
    if config.get("load_parameters") is not None:
        try:
            with open(config["load_parameters"], "r") as f:
                loaded_params = json.load(f)
            print(f"Loaded evaluation parameters from {config['load_parameters']}: {loaded_params}")
        except Exception as e:
            print(f"Failed to load parameters from {config['load_parameters']}: {e}")
            loaded_params = None
        if loaded_params is not None:
            # Construct candidate in the same order as get_optimizable_params():
            candidate = [
                loaded_params.get("profit_threshold", plugin.params["profit_threshold"]),
                loaded_params.get("tp_multiplier", plugin.params["tp_multiplier"]),
                loaded_params.get("sl_multiplier", plugin.params["sl_multiplier"]),
                loaded_params.get("lower_rr_threshold", plugin.params["lower_rr_threshold"]),
                loaded_params.get("upper_rr_threshold", plugin.params["upper_rr_threshold"]),
                int(loaded_params.get("time_horizon", 3))
            ]
            print(f"Evaluating strategy with loaded parameters: {candidate}")
            # Initialize optimizer globals so that evaluate_individual() can use them
            init_optimizer(plugin, base_data, hourly_preds, daily_preds, config)
            result = evaluate_individual(candidate)
            trading_info = {"best_parameters": {
                "profit_threshold": candidate[0],
                "tp_multiplier": candidate[1],
                "sl_multiplier": candidate[2],
                "lower_rr_threshold": candidate[3],
                "upper_rr_threshold": candidate[4],
                "time_horizon": candidate[5]
            }, "profit": result[0]}
        else:
            trading_info = {}
    else:
        # Otherwise, run full optimization.
        if hasattr(plugin, "get_optimizable_params") and hasattr(plugin, "evaluate_candidate"):
            print(f"\nPlugin supports optimization. Running optimizer for '{strat_name}'...")
            trading_info = run_optimizer(plugin, base_data, hourly_preds, daily_preds, config)
        else:
            print("\nPlugin does not support optimization. Exiting.")
            trading_info = {}

    print("\n=== Optimization Results ===")
    for key, value in trading_info.items():
        print(f"{key}: {value}")

    # Rename the final balance plot if 'balance_plot_file' is provided.
    if config.get("balance_plot_file"):
        old_plot = "balance_plot.png"
        new_plot = config["balance_plot_file"]
        if os.path.exists(old_plot):
            try:
                os.rename(old_plot, new_plot)
                print(f"Renamed {old_plot} -> {new_plot}")
            except Exception as e:
                print(f"Failed to rename {old_plot} to {new_plot}: {e}")
        else:
            print(f"Warning: {old_plot} not found; no balance plot to rename.")

    # Save trades to CSV if 'trades_csv_file' is specified.
    trades_csv = config.get("trades_csv_file")
    if trades_csv:
        try:
            if hasattr(plugin, "trades") and plugin.trades:
                pd.DataFrame(plugin.trades).to_csv(trades_csv, index=False)
                print(f"Trades saved to {trades_csv}.")
            else:
                print("Warning: plugin.trades not found or empty.")
        except Exception as e:
            print(f"Failed to save trades to {trades_csv}: {e}")

    # Save summary CSV if 'summary_csv_file' is specified.
    summary_csv = config.get("summary_csv_file")
    if summary_csv:
        try:
            df = pd.DataFrame([trading_info])
            df.to_csv(summary_csv, index=False)
            print(f"Summary saved to {summary_csv}.")
        except Exception as e:
            print(f"Failed to save summary CSV to {summary_csv}: {e}")

    # Save best parameters as JSON if in optimization mode and load_parameters is not provided.
    if config.get("load_parameters") is None and config.get("save_parameters"):
        try:
            with open(config["save_parameters"], "w") as f:
                json.dump(trading_info.get("best_parameters", {}), f, indent=4, default=str)
            print(f"Best parameters saved to {config['save_parameters']}.")
        except Exception as e:
            print(f"Failed to save best parameters to {config['save_parameters']}: {e}")

    end_time = time.time()
    print(f"\nTotal Execution Time: {end_time - start_time:.2f} seconds")
    return trading_info, getattr(plugin, "trades", None)
Con estas modificaciones se recortan ambos datasets al rango de fechas en común (sin sobreescribir ninguna fecha original) y se verifica/imprime el rango final antes de pasar los datos al plugin.





Dijiste:
parece que está bien, pero este codigo también alinea el dataset base_dataset_file? es decir se debe buscar el rango de fechas común a los 3 y se debe hacer la alineación por medio de trimings al inicio ay al final de cada uno de manera que al pasar los 3 datasets (base, preedicctiones diarias y las horarias) los tres deben coincidir en el rango de fechas y el número de filas. Por favor verifica que los 3 datasets se estén alineando correctamente y dime exactamente en que linea de código se está alineando el dataset base con los  de las predicciones
Razonado durante 11 segundos
En el código actualizado, solo se alinea el rango de fechas entre los datasets de predicciones (horarias y diarias). Esto se realiza en el bloque donde se calcula la intersección de sus índices, es decir, en las líneas:

    common_index = hourly_df.index.intersection(daily_df.index)
    if common_index.empty:
        raise ValueError("No common date range found between hourly and daily predictions.")
    hourly_df = hourly_df.loc[common_index]
    daily_df = daily_df.loc[common_index]
Estas líneas (por ejemplo, líneas 40–43 en el código actualizado) recortan hourly_df y daily_df al rango común.

Sin embargo, el dataset base (base_df) se carga y se utiliza sin recortarlo al rango común, por lo que no se está alineando con las predicciones. Es decir, en el código actual no existe una línea que recorte o alinee base_df al mismo rango de fechas que los datasets de predicciones.





Dijiste:
ok, eso está mal, por favor, asegúrate que los tres datasets estén alineados en el mismo rango de fechas(y horas), los valores de inicio y final de cada uno pasados al plugin, deben coincidir exáctamente 
import pandas as pd
import numpy as np
import time
from app.data_handler import load_csv
from app.optimizer import run_optimizer

# =============================================================================
# DATA PROCESSOR FOR TRADING STRATEGY OPTIMIZATION
# =============================================================================
#
# Processes datasets and runs the optimizer.
# Calls run_optimizer() from app.optimizer to optimize the trading strategy.
# =============================================================================

def create_hourly_predictions(df, horizon):
    """
    Auto-compute hourly predictions from the base dataset.
    """
    blocks = []
    for i in range(len(df) - horizon):
        block = df.iloc[i+1 : i+1+horizon].values.flatten()
        blocks.append(block)
    return pd.DataFrame(blocks, index=df.index[:-horizon])

def create_daily_predictions(df, horizon):
    """
    Auto-compute daily predictions from the base dataset.

    Each row i in 'df' yields a block of predicted values at offsets t+24, t+48, ... t+24*horizon.
    If there are not enough rows (or any row indexing is invalid), we return an empty DataFrame.
    """
    nrows = len(df)
    required_rows = horizon * 24

    # If the dataset is too short for daily predictions, return empty.
    if nrows < required_rows:
        print("Warning: Not enough rows to create daily predictions. Returning an empty DataFrame.")
        return pd.DataFrame()

    blocks = []
    # For each row i, gather the next horizon days (24*horizon ticks).
    for i in range(nrows - required_rows):
        block_values = []
        # Attempt to fetch each day offset at i + d*24
        for d in range(1, horizon + 1):
            idx = i + d * 24
            # If idx is out of range for some reason (edge cases), skip safely
            if idx >= nrows:
                break
            row_vals = df.iloc[idx].values  # shape: (features,)
            if row_vals.ndim == 0:
                # means we have an empty row or zero-dimensional array
                continue
            block_values.extend(row_vals.flatten())
        # If we got some values, append them to blocks
        if block_values:
            blocks.append(block_values)
        else:
            # If we ended up with no valid data for that block, skip
            continue

    if not blocks:
        # No valid blocks were created
        print("Warning: daily predictions are empty after alignment. Returning an empty DataFrame.")
        return pd.DataFrame()

    # The resulting blocks length = nrows - required_rows
    # Align the index accordingly
    daily_idx = df.index[:(nrows - required_rows)]
    return pd.DataFrame(blocks, index=daily_idx)



def process_data(config):
    """
    Loads and processes datasets, ensuring alignment and applying max_steps.
    - Uses external prediction files if provided.
    - Generates predictions if files are not available.
    - Ensures all datasets are properly aligned to the common date range.
    """
    import pandas as pd
    import numpy as np
    import json
    from app.data_handler import load_csv

    headers = config.get("headers", True)
    print("Loading datasets...")

    # Load datasets based on config parameters
    hourly_df = load_csv(config["hourly_predictions_file"], headers=headers) if config.get("hourly_predictions_file") else None
    daily_df = load_csv(config["daily_predictions_file"], headers=headers) if config.get("daily_predictions_file") else None
    base_df = load_csv(config["base_dataset_file"], headers=headers)

    print(f"Base dataset loaded: {base_df.shape}")

    # Auto-generate predictions if files are missing
    if hourly_df is None:
        if "time_horizon" not in config or not config["time_horizon"]:
            raise ValueError("time_horizon must be provided when auto-generating predictions.")
        print("Auto-generating hourly predictions...")
        hourly_df = create_hourly_predictions(base_df, config["time_horizon"])

    if daily_df is None:
        if "time_horizon" not in config or not config["time_horizon"]:
            raise ValueError("time_horizon must be provided when auto-generating predictions.")
        print("Auto-generating daily predictions...")
        daily_df = create_daily_predictions(base_df, config["time_horizon"])

    # Ensure that hourly and daily predictions have a datetime index based on DATE_TIME column, if not already set.
    if hourly_df is not None:
        if not isinstance(hourly_df.index, pd.DatetimeIndex):
            if "DATE_TIME" in hourly_df.columns:
                hourly_df.index = pd.to_datetime(hourly_df["DATE_TIME"])
            else:
                raise ValueError("hourly_df does not have a DATE_TIME column.")
    if daily_df is not None:
        if not isinstance(daily_df.index, pd.DatetimeIndex):
            if "DATE_TIME" in daily_df.columns:
                daily_df.index = pd.to_datetime(daily_df["DATE_TIME"])
            else:
                raise ValueError("daily_df does not have a DATE_TIME column.")

    # Align datasets to the common date range (intersection of their datetime indexes)
    common_index = hourly_df.index.intersection(daily_df.index)
    if common_index.empty:
        raise ValueError("No common date range found between hourly and daily predictions.")
    hourly_df = hourly_df.loc[common_index]
    daily_df = daily_df.loc[common_index]

    # Verify that both datasets have the same number of rows
    if len(hourly_df) != len(daily_df):
        raise ValueError("After alignment, hourly and daily predictions have different number of rows!")

    # Print aligned date ranges
    print(f"Aligned Hourly predictions range: {hourly_df.index.min()} to {hourly_df.index.max()}")
    print(f"Aligned Daily predictions range: {daily_df.index.min()} to {daily_df.index.max()}")

    return {"hourly": hourly_df, "daily": daily_df, "base": base_df}


def run_processing_pipeline(config, plugin):
    """
    Executes the trading strategy optimization pipeline.
    
    - Loads and processes datasets.
    - If config["load_parameters"] is provided (not None), loads candidate parameters from the specified JSON file
      and evaluates the strategy once using those parameters (printing all trades, summary, and saving outputs).
    - Otherwise, runs the full optimization via run_optimizer().
    - At the end, renames the balance plot and saves the trades and summary CSV files.
    - If in optimization mode and if config["save_parameters"] is provided, saves the best parameters as JSON.
    """
    import json, os, pandas as pd
    from app.optimizer import init_optimizer, evaluate_individual, run_optimizer
    import time

    start_time = time.time()
    strat_name = config.get("strategy_name", "Heuristic Strategy")
    print(f"\n=== Starting Trading Strategy Optimization Pipeline for '{strat_name}' ===")

    datasets = process_data(config)
    hourly_preds, daily_preds, base_data = datasets["hourly"], datasets["daily"], datasets["base"]

    # Additional verification: print final aligned date ranges before passing data to the plugin
    print(f"Final Hourly predictions date range: {hourly_preds.index.min()} to {hourly_preds.index.max()}")
    print(f"Final Daily predictions date range: {daily_preds.index.min()} to {daily_preds.index.max()}")

    print("\nProcessed Dataset Shapes:")
    print(f"  Hourly predictions: {hourly_preds.shape}")
    print(f"  Daily predictions:  {daily_preds.shape}")
    print(f"  Base dataset:       {base_data.shape}")

    # If load_parameters is provided, load candidate parameters and evaluate the strategy once.
    if config.get("load_parameters") is not None:
        try:
            with open(config["load_parameters"], "r") as f:
                loaded_params = json.load(f)
            print(f"Loaded evaluation parameters from {config['load_parameters']}: {loaded_params}")
        except Exception as e:
            print(f"Failed to load parameters from {config['load_parameters']}: {e}")
            loaded_params = None
        if loaded_params is not None:
            # Construct candidate in the same order as get_optimizable_params():
            candidate = [
                loaded_params.get("profit_threshold", plugin.params["profit_threshold"]),
                loaded_params.get("tp_multiplier", plugin.params["tp_multiplier"]),
                loaded_params.get("sl_multiplier", plugin.params["sl_multiplier"]),
                loaded_params.get("lower_rr_threshold", plugin.params["lower_rr_threshold"]),
                loaded_params.get("upper_rr_threshold", plugin.params["upper_rr_threshold"]),
                int(loaded_params.get("time_horizon", 3))
            ]
            print(f"Evaluating strategy with loaded parameters: {candidate}")
            # Initialize optimizer globals so that evaluate_individual() can use them
            init_optimizer(plugin, base_data, hourly_preds, daily_preds, config)
            result = evaluate_individual(candidate)
            trading_info = {"best_parameters": {
                "profit_threshold": candidate[0],
                "tp_multiplier": candidate[1],
                "sl_multiplier": candidate[2],
                "lower_rr_threshold": candidate[3],
                "upper_rr_threshold": candidate[4],
                "time_horizon": candidate[5]
            }, "profit": result[0]}
        else:
            trading_info = {}
    else:
        # Otherwise, run full optimization.
        if hasattr(plugin, "get_optimizable_params") and hasattr(plugin, "evaluate_candidate"):
            print(f"\nPlugin supports optimization. Running optimizer for '{strat_name}'...")
            trading_info = run_optimizer(plugin, base_data, hourly_preds, daily_preds, config)
        else:
            print("\nPlugin does not support optimization. Exiting.")
            trading_info = {}

    print("\n=== Optimization Results ===")
    for key, value in trading_info.items():
        print(f"{key}: {value}")

    # Rename the final balance plot if 'balance_plot_file' is provided.
    if config.get("balance_plot_file"):
        old_plot = "balance_plot.png"
        new_plot = config["balance_plot_file"]
        if os.path.exists(old_plot):
            try:
                os.rename(old_plot, new_plot)
                print(f"Renamed {old_plot} -> {new_plot}")
            except Exception as e:
                print(f"Failed to rename {old_plot} to {new_plot}: {e}")
        else:
            print(f"Warning: {old_plot} not found; no balance plot to rename.")

    # Save trades to CSV if 'trades_csv_file' is specified.
    trades_csv = config.get("trades_csv_file")
    if trades_csv:
        try:
            if hasattr(plugin, "trades") and plugin.trades:
                pd.DataFrame(plugin.trades).to_csv(trades_csv, index=False)
                print(f"Trades saved to {trades_csv}.")
            else:
                print("Warning: plugin.trades not found or empty.")
        except Exception as e:
            print(f"Failed to save trades to {trades_csv}: {e}")

    # Save summary CSV if 'summary_csv_file' is specified.
    summary_csv = config.get("summary_csv_file")
    if summary_csv:
        try:
            df = pd.DataFrame([trading_info])
            df.to_csv(summary_csv, index=False)
            print(f"Summary saved to {summary_csv}.")
        except Exception as e:
            print(f"Failed to save summary CSV to {summary_csv}: {e}")

    # Save best parameters as JSON if in optimization mode and load_parameters is not provided.
    if config.get("load_parameters") is None and config.get("save_parameters"):
        try:
            with open(config["save_parameters"], "w") as f:
                json.dump(trading_info.get("best_parameters", {}), f, indent=4, default=str)
            print(f"Best parameters saved to {config['save_parameters']}.")
        except Exception as e:
            print(f"Failed to save best parameters to {config['save_parameters']}: {e}")

    end_time = time.time()
    print(f"\nTotal Execution Time: {end_time - start_time:.2f} seconds")
    return trading_info, getattr(plugin, "trades", None)


def run_processing_pipeline(config, plugin):
    """
    Executes the trading strategy optimization pipeline.
    
    - Loads and processes datasets.
    - If config["load_parameters"] is provided (not None), loads candidate parameters from the specified JSON file
      and evaluates the strategy once using those parameters (printing all trades, summary, and saving outputs).
    - Otherwise, runs the full optimization via run_optimizer().
    - At the end, renames the balance plot and saves the trades and summary CSV files.
    - If in optimization mode and if config["save_parameters"] is provided, saves the best parameters as JSON.
    """
    import json, os, pandas as pd
    from app.optimizer import init_optimizer, evaluate_individual, run_optimizer
    import time

    start_time = time.time()
    strat_name = config.get("strategy_name", "Heuristic Strategy")
    print(f"\n=== Starting Trading Strategy Optimization Pipeline for '{strat_name}' ===")

    datasets = process_data(config)
    hourly_preds, daily_preds, base_data = datasets["hourly"], datasets["daily"], datasets["base"]

    print("\nProcessed Dataset Shapes:")
    print(f"  Hourly predictions: {hourly_preds.shape}")
    print(f"  Daily predictions:  {daily_preds.shape}")
    print(f"  Base dataset:       {base_data.shape}")

    # If load_parameters is provided, load candidate parameters and evaluate the strategy once.
    if config.get("load_parameters") is not None:
        try:
            with open(config["load_parameters"], "r") as f:
                loaded_params = json.load(f)
            print(f"Loaded evaluation parameters from {config['load_parameters']}: {loaded_params}")
        except Exception as e:
            print(f"Failed to load parameters from {config['load_parameters']}: {e}")
            loaded_params = None
        if loaded_params is not None:
            # Construct candidate in the same order as get_optimizable_params():
            candidate = [
                loaded_params.get("profit_threshold", plugin.params["profit_threshold"]),
                loaded_params.get("tp_multiplier", plugin.params["tp_multiplier"]),
                loaded_params.get("sl_multiplier", plugin.params["sl_multiplier"]),
                loaded_params.get("lower_rr_threshold", plugin.params["lower_rr_threshold"]),
                loaded_params.get("upper_rr_threshold", plugin.params["upper_rr_threshold"]),
                int(loaded_params.get("time_horizon", 3))
            ]
            print(f"Evaluating strategy with loaded parameters: {candidate}")
            # Initialize optimizer globals so that evaluate_individual() can use them
            init_optimizer(plugin, base_data, hourly_preds, daily_preds, config)
            result = evaluate_individual(candidate)
            trading_info = {"best_parameters": {
                "profit_threshold": candidate[0],
                "tp_multiplier": candidate[1],
                "sl_multiplier": candidate[2],
                "lower_rr_threshold": candidate[3],
                "upper_rr_threshold": candidate[4],
                "time_horizon": candidate[5]
            }, "profit": result[0]}
        else:
            trading_info = {}
    else:
        # Otherwise, run full optimization.
        if hasattr(plugin, "get_optimizable_params") and hasattr(plugin, "evaluate_candidate"):
            print(f"\nPlugin supports optimization. Running optimizer for '{strat_name}'...")
            trading_info = run_optimizer(plugin, base_data, hourly_preds, daily_preds, config)
        else:
            print("\nPlugin does not support optimization. Exiting.")
            trading_info = {}

    print("\n=== Optimization Results ===")
    for key, value in trading_info.items():
        print(f"{key}: {value}")

    # Rename the final balance plot if 'balance_plot_file' is provided.
    if config.get("balance_plot_file"):
        old_plot = "balance_plot.png"
        new_plot = config["balance_plot_file"]
        if os.path.exists(old_plot):
            try:
                os.rename(old_plot, new_plot)
                print(f"Renamed {old_plot} -> {new_plot}")
            except Exception as e:
                print(f"Failed to rename {old_plot} to {new_plot}: {e}")
        else:
            print(f"Warning: {old_plot} not found; no balance plot to rename.")

    # Save trades to CSV if 'trades_csv_file' is specified.
    trades_csv = config.get("trades_csv_file")
    if trades_csv:
        try:
            if hasattr(plugin, "trades") and plugin.trades:
                pd.DataFrame(plugin.trades).to_csv(trades_csv, index=False)
                print(f"Trades saved to {trades_csv}.")
            else:
                print("Warning: plugin.trades not found or empty.")
        except Exception as e:
            print(f"Failed to save trades to {trades_csv}: {e}")

    # Save summary CSV if 'summary_csv_file' is specified.
    summary_csv = config.get("summary_csv_file")
    if summary_csv:
        try:
            df = pd.DataFrame([trading_info])
            df.to_csv(summary_csv, index=False)
            print(f"Summary saved to {summary_csv}.")
        except Exception as e:
            print(f"Failed to save summary CSV to {summary_csv}: {e}")

    # Save best parameters as JSON if in optimization mode and load_parameters is not provided.
    if config.get("load_parameters") is None and config.get("save_parameters"):
        try:
            with open(config["save_parameters"], "w") as f:
                json.dump(trading_info.get("best_parameters", {}), f, indent=4, default=str)
            print(f"Best parameters saved to {config['save_parameters']}.")
        except Exception as e:
            print(f"Failed to save best parameters to {config['save_parameters']}: {e}")

    end_time = time.time()
    print(f"\nTotal Execution Time: {end_time - start_time:.2f} seconds")
    return trading_info, getattr(plugin, "trades", None)



if __name__ == "__main__":
    pass
Razonado durante 15 segundos
A continuación te muestro el código modificado para que los tres datasets (base, predicciones horarias y predicciones diarias) se recorten al rango de fechas y horas común. En este nuevo código se verifica que cada dataset tenga un índice datetime (convertido a partir de la columna "DATE_TIME" si es necesario) y luego se calcula la intersección de los índices de los tres datasets. Esa intersección se usa para "trimar" cada uno de ellos.

En concreto, la alineación del dataset base se realiza en la siguiente línea dentro de la función process_data:

base_df = base_df.loc[common_index]
Esta línea (junto con las asignaciones para las predicciones) se encuentra en el bloque donde se calcula el índice común (common_index). Aquí tienes el código completo actualizado para los métodos que requieren cambio:

def process_data(config):
    """
    Loads and processes datasets, ensuring alignment and applying max_steps.
    - Uses external prediction files if provided.
    - Generates predictions if files are not available.
    - Ensures all datasets (base, hourly predictions, daily predictions) are properly aligned to the common date range.
    """
    import pandas as pd
    import numpy as np
    import json
    from app.data_handler import load_csv

    headers = config.get("headers", True)
    print("Loading datasets...")

    # Load datasets based on config parameters
    hourly_df = load_csv(config["hourly_predictions_file"], headers=headers) if config.get("hourly_predictions_file") else None
    daily_df = load_csv(config["daily_predictions_file"], headers=headers) if config.get("daily_predictions_file") else None
    base_df = load_csv(config["base_dataset_file"], headers=headers)

    print(f"Base dataset loaded: {base_df.shape}")

    # Auto-generate predictions if files are missing
    if hourly_df is None:
        if "time_horizon" not in config or not config["time_horizon"]:
            raise ValueError("time_horizon must be provided when auto-generating predictions.")
        print("Auto-generating hourly predictions...")
        hourly_df = create_hourly_predictions(base_df, config["time_horizon"])

    if daily_df is None:
        if "time_horizon" not in config or not config["time_horizon"]:
            raise ValueError("time_horizon must be provided when auto-generating predictions.")
        print("Auto-generating daily predictions...")
        daily_df = create_daily_predictions(base_df, config["time_horizon"])

    # Ensure that hourly and daily predictions have a datetime index based on DATE_TIME column, if not already set.
    if hourly_df is not None:
        if not isinstance(hourly_df.index, pd.DatetimeIndex):
            if "DATE_TIME" in hourly_df.columns:
                hourly_df.index = pd.to_datetime(hourly_df["DATE_TIME"])
            else:
                raise ValueError("hourly_df does not have a DATE_TIME column.")
    if daily_df is not None:
        if not isinstance(daily_df.index, pd.DatetimeIndex):
            if "DATE_TIME" in daily_df.columns:
                daily_df.index = pd.to_datetime(daily_df["DATE_TIME"])
            else:
                raise ValueError("daily_df does not have a DATE_TIME column.")

    # Ensure base_df has a datetime index as well.
    if not isinstance(base_df.index, pd.DatetimeIndex):
        if "DATE_TIME" in base_df.columns:
            base_df.index = pd.to_datetime(base_df["DATE_TIME"])
        else:
            raise ValueError("base_df does not have a DATE_TIME column.")

    # Align all datasets to the common date range (intersection of their datetime indexes)
    common_index = base_df.index.intersection(hourly_df.index).intersection(daily_df.index)
    if common_index.empty:
        raise ValueError("No common date range found among base, hourly, and daily predictions.")

    # Aquí se recortan los tres datasets al rango común
    base_df = base_df.loc[common_index]         # <-- Alinea el dataset base
    hourly_df = hourly_df.loc[common_index]
    daily_df = daily_df.loc[common_index]

    # Verify that all datasets have the same number of rows
    if not (len(base_df) == len(hourly_df) == len(daily_df)):
        raise ValueError("After alignment, the number of rows in base, hourly, and daily predictions do not match!")

    # Print aligned date ranges
    print(f"Aligned Base dataset range: {base_df.index.min()} to {base_df.index.max()}")
    print(f"Aligned Hourly predictions range: {hourly_df.index.min()} to {hourly_df.index.max()}")
    print(f"Aligned Daily predictions range: {daily_df.index.min()} to {daily_df.index.max()}")

    return {"hourly": hourly_df, "daily": daily_df, "base": base_df}


def run_processing_pipeline(config, plugin):
    """
    Executes the trading strategy optimization pipeline.
    
    - Loads and processes datasets.
    - If config["load_parameters"] is provided, loads candidate parameters from the specified JSON file and evaluates the strategy once.
    - Otherwise, runs the full optimization via run_optimizer().
    - At the end, renames the balance plot and saves the trades and summary CSV files.
    - If in optimization mode and config["save_parameters"] is provided, saves the best parameters as JSON.
    """
    import json, os, pandas as pd
    from app.optimizer import init_optimizer, evaluate_individual, run_optimizer
    import time

    start_time = time.time()
    strat_name = config.get("strategy_name", "Heuristic Strategy")
    print(f"\n=== Starting Trading Strategy Optimization Pipeline for '{strat_name}' ===")

    datasets = process_data(config)
    hourly_preds, daily_preds, base_data = datasets["hourly"], datasets["daily"], datasets["base"]

    # Additional verification: print final aligned date ranges before passing data to the plugin
    print(f"Final Base dataset date range: {base_data.index.min()} to {base_data.index.max()}")
    print(f"Final Hourly predictions date range: {hourly_preds.index.min()} to {hourly_preds.index.max()}")
    print(f"Final Daily predictions date range: {daily_preds.index.min()} to {daily_preds.index.max()}")

    print("\nProcessed Dataset Shapes:")
    print(f"  Base dataset:       {base_data.shape}")
    print(f"  Hourly predictions: {hourly_preds.shape}")
    print(f"  Daily predictions:  {daily_preds.shape}")

    # If load_parameters is provided, load candidate parameters and evaluate the strategy once.
    if config.get("load_parameters") is not None:
        try:
            with open(config["load_parameters"], "r") as f:
                loaded_params = json.load(f)
            print(f"Loaded evaluation parameters from {config['load_parameters']}: {loaded_params}")
        except Exception as e:
            print(f"Failed to load parameters from {config['load_parameters']}: {e}")
            loaded_params = None
        if loaded_params is not None:
            # Construct candidate in the same order as get_optimizable_params():
            candidate = [
                loaded_params.get("profit_threshold", plugin.params["profit_threshold"]),
                loaded_params.get("tp_multiplier", plugin.params["tp_multiplier"]),
                loaded_params.get("sl_multiplier", plugin.params["sl_multiplier"]),
                loaded_params.get("lower_rr_threshold", plugin.params["lower_rr_threshold"]),
                loaded_params.get("upper_rr_threshold", plugin.params["upper_rr_threshold"]),
                int(loaded_params.get("time_horizon", 3))
            ]
            print(f"Evaluating strategy with loaded parameters: {candidate}")
            # Initialize optimizer globals so that evaluate_individual() can use them
            init_optimizer(plugin, base_data, hourly_preds, daily_preds, config)
            result = evaluate_individual(candidate)
            trading_info = {"best_parameters": {
                "profit_threshold": candidate[0],
                "tp_multiplier": candidate[1],
                "sl_multiplier": candidate[2],
                "lower_rr_threshold": candidate[3],
                "upper_rr_threshold": candidate[4],
                "time_horizon": candidate[5]
            }, "profit": result[0]}
        else:
            trading_info = {}
    else:
        # Otherwise, run full optimization.
        if hasattr(plugin, "get_optimizable_params") and hasattr(plugin, "evaluate_candidate"):
            print(f"\nPlugin supports optimization. Running optimizer for '{strat_name}'...")
            trading_info = run_optimizer(plugin, base_data, hourly_preds, daily_preds, config)
        else:
            print("\nPlugin does not support optimization. Exiting.")
            trading_info = {}

    print("\n=== Optimization Results ===")
    for key, value in trading_info.items():
        print(f"{key}: {value}")

    # Rename the final balance plot if 'balance_plot_file' is provided.
    if config.get("balance_plot_file"):
        old_plot = "balance_plot.png"
        new_plot = config["balance_plot_file"]
        if os.path.exists(old_plot):
            try:
                os.rename(old_plot, new_plot)
                print(f"Renamed {old_plot} -> {new_plot}")
            except Exception as e:
                print(f"Failed to rename {old_plot} to {new_plot}: {e}")
        else:
            print(f"Warning: {old_plot} not found; no balance plot to rename.")

    # Save trades to CSV if 'trades_csv_file' is specified.
    trades_csv = config.get("trades_csv_file")
    if trades_csv:
        try:
            if hasattr(plugin, "trades") and plugin.trades:
                pd.DataFrame(plugin.trades).to_csv(trades_csv, index=False)
                print(f"Trades saved to {trades_csv}.")
            else:
                print("Warning: plugin.trades not found or empty.")
        except Exception as e:
            print(f"Failed to save trades to {trades_csv}: {e}")

    # Save summary CSV if 'summary_csv_file' is specified.
    summary_csv = config.get("summary_csv_file")
    if summary_csv:
        try:
            df = pd.DataFrame([trading_info])
            df.to_csv(summary_csv, index=False)
            print(f"Summary saved to {summary_csv}.")
        except Exception as e:
            print(f"Failed to save summary CSV to {summary_csv}: {e}")

    # Save best parameters as JSON if in optimization mode and load_parameters is not provided.
    if config.get("load_parameters") is None and config.get("save_parameters"):
        try:
            with open(config["save_parameters"], "w") as f:
                json.dump(trading_info.get("best_parameters", {}), f, indent=4, default=str)
            print(f"Best parameters saved to {config['save_parameters']}.")
        except Exception as e:
            print(f"Failed to save best parameters to {config['save_parameters']}: {e}")

    end_time = time.time()
    print(f"\nTotal Execution Time: {end_time - start_time:.2f} seconds")
    return trading_info, getattr(plugin, "trades", None)
Con estas modificaciones, los tres datasets se recortan al rango común de fechas (y horas) y se garantiza que tengan el mismo número de filas al pasarlos al plugin. En particular, la línea que alinea el dataset base es:

base_df = base_df.loc[common_index]




Dijiste:
Ok, ahora como sabemos que el dataset de predicciones horarias contiene las predicciones para las siguientes (número de columnas) horas y el dataset de predicciones diarias tiene las predicciones para los siguientes (número de sus columnas) dias,  y dado que tenemos el dataset base, una vez los datasets han sido alineados, por favor calcula el MAE y el R^2 para cada una de las predicciones (cada una de las horarias y de las diarias) respecto al valor de la columna CLOSE del dataset base que contiene. Imprimelos en una tabla antes de enviar los datos al plugin de estrategia.

import pandas as pd
import numpy as np
import time
from app.data_handler import load_csv
from app.optimizer import run_optimizer

# =============================================================================
# DATA PROCESSOR FOR TRADING STRATEGY OPTIMIZATION
# =============================================================================
#
# Processes datasets and runs the optimizer.
# Calls run_optimizer() from app.optimizer to optimize the trading strategy.
# =============================================================================

def create_hourly_predictions(df, horizon):
    """
    Auto-compute hourly predictions from the base dataset.
    """
    blocks = []
    for i in range(len(df) - horizon):
        block = df.iloc[i+1 : i+1+horizon].values.flatten()
        blocks.append(block)
    return pd.DataFrame(blocks, index=df.index[:-horizon])

def create_daily_predictions(df, horizon):
    """
    Auto-compute daily predictions from the base dataset.

    Each row i in 'df' yields a block of predicted values at offsets t+24, t+48, ... t+24*horizon.
    If there are not enough rows (or any row indexing is invalid), we return an empty DataFrame.
    """
    nrows = len(df)
    required_rows = horizon * 24

    # If the dataset is too short for daily predictions, return empty.
    if nrows < required_rows:
        print("Warning: Not enough rows to create daily predictions. Returning an empty DataFrame.")
        return pd.DataFrame()

    blocks = []
    # For each row i, gather the next horizon days (24*horizon ticks).
    for i in range(nrows - required_rows):
        block_values = []
        # Attempt to fetch each day offset at i + d*24
        for d in range(1, horizon + 1):
            idx = i + d * 24
            # If idx is out of range for some reason (edge cases), skip safely
            if idx >= nrows:
                break
            row_vals = df.iloc[idx].values  # shape: (features,)
            if row_vals.ndim == 0:
                # means we have an empty row or zero-dimensional array
                continue
            block_values.extend(row_vals.flatten())
        # If we got some values, append them to blocks
        if block_values:
            blocks.append(block_values)
        else:
            # If we ended up with no valid data for that block, skip
            continue

    if not blocks:
        # No valid blocks were created
        print("Warning: daily predictions are empty after alignment. Returning an empty DataFrame.")
        return pd.DataFrame()

    # The resulting blocks length = nrows - required_rows
    # Align the index accordingly
    daily_idx = df.index[:(nrows - required_rows)]
    return pd.DataFrame(blocks, index=daily_idx)



def process_data(config):
    """
    Loads and processes datasets, ensuring alignment and applying max_steps.
    - Uses external prediction files if provided.
    - Generates predictions if files are not available.
    - Ensures all datasets (base, hourly predictions, daily predictions) are properly aligned to the common date range.
    """
    import pandas as pd
    import numpy as np
    import json
    from app.data_handler import load_csv

    headers = config.get("headers", True)
    print("Loading datasets...")

    # Load datasets based on config parameters
    hourly_df = load_csv(config["hourly_predictions_file"], headers=headers) if config.get("hourly_predictions_file") else None
    daily_df = load_csv(config["daily_predictions_file"], headers=headers) if config.get("daily_predictions_file") else None
    base_df = load_csv(config["base_dataset_file"], headers=headers)

    print(f"Base dataset loaded: {base_df.shape}")

    # Auto-generate predictions if files are missing
    if hourly_df is None:
        if "time_horizon" not in config or not config["time_horizon"]:
            raise ValueError("time_horizon must be provided when auto-generating predictions.")
        print("Auto-generating hourly predictions...")
        hourly_df = create_hourly_predictions(base_df, config["time_horizon"])

    if daily_df is None:
        if "time_horizon" not in config or not config["time_horizon"]:
            raise ValueError("time_horizon must be provided when auto-generating predictions.")
        print("Auto-generating daily predictions...")
        daily_df = create_daily_predictions(base_df, config["time_horizon"])

    # Ensure that hourly and daily predictions have a datetime index based on DATE_TIME column, if not already set.
    if hourly_df is not None:
        if not isinstance(hourly_df.index, pd.DatetimeIndex):
            if "DATE_TIME" in hourly_df.columns:
                hourly_df.index = pd.to_datetime(hourly_df["DATE_TIME"])
            else:
                raise ValueError("hourly_df does not have a DATE_TIME column.")
    if daily_df is not None:
        if not isinstance(daily_df.index, pd.DatetimeIndex):
            if "DATE_TIME" in daily_df.columns:
                daily_df.index = pd.to_datetime(daily_df["DATE_TIME"])
            else:
                raise ValueError("daily_df does not have a DATE_TIME column.")

    # Ensure base_df has a datetime index as well.
    if not isinstance(base_df.index, pd.DatetimeIndex):
        if "DATE_TIME" in base_df.columns:
            base_df.index = pd.to_datetime(base_df["DATE_TIME"])
        else:
            raise ValueError("base_df does not have a DATE_TIME column.")

    # Align all datasets to the common date range (intersection of their datetime indexes)
    common_index = base_df.index.intersection(hourly_df.index).intersection(daily_df.index)
    if common_index.empty:
        raise ValueError("No common date range found among base, hourly, and daily predictions.")

    # Aquí se recortan los tres datasets al rango común
    base_df = base_df.loc[common_index]         # <-- Alinea el dataset base
    hourly_df = hourly_df.loc[common_index]
    daily_df = daily_df.loc[common_index]

    # Verify that all datasets have the same number of rows
    if not (len(base_df) == len(hourly_df) == len(daily_df)):
        raise ValueError("After alignment, the number of rows in base, hourly, and daily predictions do not match!")

    # Print aligned date ranges
    print(f"Aligned Base dataset range: {base_df.index.min()} to {base_df.index.max()}")
    print(f"Aligned Hourly predictions range: {hourly_df.index.min()} to {hourly_df.index.max()}")
    print(f"Aligned Daily predictions range: {daily_df.index.min()} to {daily_df.index.max()}")

    return {"hourly": hourly_df, "daily": daily_df, "base": base_df}


def run_processing_pipeline(config, plugin):
    """
    Executes the trading strategy optimization pipeline.
    
    - Loads and processes datasets.
    - If config["load_parameters"] is provided, loads candidate parameters from the specified JSON file and evaluates the strategy once.
    - Otherwise, runs the full optimization via run_optimizer().
    - At the end, renames the balance plot and saves the trades and summary CSV files.
    - If in optimization mode and config["save_parameters"] is provided, saves the best parameters as JSON.
    """
    import json, os, pandas as pd
    from app.optimizer import init_optimizer, evaluate_individual, run_optimizer
    import time

    start_time = time.time()
    strat_name = config.get("strategy_name", "Heuristic Strategy")
    print(f"\n=== Starting Trading Strategy Optimization Pipeline for '{strat_name}' ===")

    datasets = process_data(config)
    hourly_preds, daily_preds, base_data = datasets["hourly"], datasets["daily"], datasets["base"]

    # Additional verification: print final aligned date ranges before passing data to the plugin
    print(f"Final Base dataset date range: {base_data.index.min()} to {base_data.index.max()}")
    print(f"Final Hourly predictions date range: {hourly_preds.index.min()} to {hourly_preds.index.max()}")
    print(f"Final Daily predictions date range: {daily_preds.index.min()} to {daily_preds.index.max()}")

    print("\nProcessed Dataset Shapes:")
    print(f"  Base dataset:       {base_data.shape}")
    print(f"  Hourly predictions: {hourly_preds.shape}")
    print(f"  Daily predictions:  {daily_preds.shape}")

    # If load_parameters is provided, load candidate parameters and evaluate the strategy once.
    if config.get("load_parameters") is not None:
        try:
            with open(config["load_parameters"], "r") as f:
                loaded_params = json.load(f)
            print(f"Loaded evaluation parameters from {config['load_parameters']}: {loaded_params}")
        except Exception as e:
            print(f"Failed to load parameters from {config['load_parameters']}: {e}")
            loaded_params = None
        if loaded_params is not None:
            # Construct candidate in the same order as get_optimizable_params():
            candidate = [
                loaded_params.get("profit_threshold", plugin.params["profit_threshold"]),
                loaded_params.get("tp_multiplier", plugin.params["tp_multiplier"]),
                loaded_params.get("sl_multiplier", plugin.params["sl_multiplier"]),
                loaded_params.get("lower_rr_threshold", plugin.params["lower_rr_threshold"]),
                loaded_params.get("upper_rr_threshold", plugin.params["upper_rr_threshold"]),
                int(loaded_params.get("time_horizon", 3))
            ]
            print(f"Evaluating strategy with loaded parameters: {candidate}")
            # Initialize optimizer globals so that evaluate_individual() can use them
            init_optimizer(plugin, base_data, hourly_preds, daily_preds, config)
            result = evaluate_individual(candidate)
            trading_info = {"best_parameters": {
                "profit_threshold": candidate[0],
                "tp_multiplier": candidate[1],
                "sl_multiplier": candidate[2],
                "lower_rr_threshold": candidate[3],
                "upper_rr_threshold": candidate[4],
                "time_horizon": candidate[5]
            }, "profit": result[0]}
        else:
            trading_info = {}
    else:
        # Otherwise, run full optimization.
        if hasattr(plugin, "get_optimizable_params") and hasattr(plugin, "evaluate_candidate"):
            print(f"\nPlugin supports optimization. Running optimizer for '{strat_name}'...")
            trading_info = run_optimizer(plugin, base_data, hourly_preds, daily_preds, config)
        else:
            print("\nPlugin does not support optimization. Exiting.")
            trading_info = {}

    print("\n=== Optimization Results ===")
    for key, value in trading_info.items():
        print(f"{key}: {value}")

    # Rename the final balance plot if 'balance_plot_file' is provided.
    if config.get("balance_plot_file"):
        old_plot = "balance_plot.png"
        new_plot = config["balance_plot_file"]
        if os.path.exists(old_plot):
            try:
                os.rename(old_plot, new_plot)
                print(f"Renamed {old_plot} -> {new_plot}")
            except Exception as e:
                print(f"Failed to rename {old_plot} to {new_plot}: {e}")
        else:
            print(f"Warning: {old_plot} not found; no balance plot to rename.")

    # Save trades to CSV if 'trades_csv_file' is specified.
    trades_csv = config.get("trades_csv_file")
    if trades_csv:
        try:
            if hasattr(plugin, "trades") and plugin.trades:
                pd.DataFrame(plugin.trades).to_csv(trades_csv, index=False)
                print(f"Trades saved to {trades_csv}.")
            else:
                print("Warning: plugin.trades not found or empty.")
        except Exception as e:
            print(f"Failed to save trades to {trades_csv}: {e}")

    # Save summary CSV if 'summary_csv_file' is specified.
    summary_csv = config.get("summary_csv_file")
    if summary_csv:
        try:
            df = pd.DataFrame([trading_info])
            df.to_csv(summary_csv, index=False)
            print(f"Summary saved to {summary_csv}.")
        except Exception as e:
            print(f"Failed to save summary CSV to {summary_csv}: {e}")

    # Save best parameters as JSON if in optimization mode and load_parameters is not provided.
    if config.get("load_parameters") is None and config.get("save_parameters"):
        try:
            with open(config["save_parameters"], "w") as f:
                json.dump(trading_info.get("best_parameters", {}), f, indent=4, default=str)
            print(f"Best parameters saved to {config['save_parameters']}.")
        except Exception as e:
            print(f"Failed to save best parameters to {config['save_parameters']}: {e}")

    end_time = time.time()
    print(f"\nTotal Execution Time: {end_time - start_time:.2f} seconds")
    return trading_info, getattr(plugin, "trades", None)


def run_processing_pipeline(config, plugin):
    """
    Executes the trading strategy optimization pipeline.
    
    - Loads and processes datasets.
    - If config["load_parameters"] is provided (not None), loads candidate parameters from the specified JSON file
      and evaluates the strategy once using those parameters (printing all trades, summary, and saving outputs).
    - Otherwise, runs the full optimization via run_optimizer().
    - At the end, renames the balance plot and saves the trades and summary CSV files.
    - If in optimization mode and if config["save_parameters"] is provided, saves the best parameters as JSON.
    """
    import json, os, pandas as pd
    from app.optimizer import init_optimizer, evaluate_individual, run_optimizer
    import time

    start_time = time.time()
    strat_name = config.get("strategy_name", "Heuristic Strategy")
    print(f"\n=== Starting Trading Strategy Optimization Pipeline for '{strat_name}' ===")

    datasets = process_data(config)
    hourly_preds, daily_preds, base_data = datasets["hourly"], datasets["daily"], datasets["base"]

    # Additional verification: print final aligned date ranges before passing data to the plugin
    print(f"Final Hourly predictions date range: {hourly_preds.index.min()} to {hourly_preds.index.max()}")
    print(f"Final Daily predictions date range: {daily_preds.index.min()} to {daily_preds.index.max()}")

    print("\nProcessed Dataset Shapes:")
    print(f"  Hourly predictions: {hourly_preds.shape}")
    print(f"  Daily predictions:  {daily_preds.shape}")
    print(f"  Base dataset:       {base_data.shape}")

    # If load_parameters is provided, load candidate parameters and evaluate the strategy once.
    if config.get("load_parameters") is not None:
        try:
            with open(config["load_parameters"], "r") as f:
                loaded_params = json.load(f)
            print(f"Loaded evaluation parameters from {config['load_parameters']}: {loaded_params}")
        except Exception as e:
            print(f"Failed to load parameters from {config['load_parameters']}: {e}")
            loaded_params = None
        if loaded_params is not None:
            # Construct candidate in the same order as get_optimizable_params():
            candidate = [
                loaded_params.get("profit_threshold", plugin.params["profit_threshold"]),
                loaded_params.get("tp_multiplier", plugin.params["tp_multiplier"]),
                loaded_params.get("sl_multiplier", plugin.params["sl_multiplier"]),
                loaded_params.get("lower_rr_threshold", plugin.params["lower_rr_threshold"]),
                loaded_params.get("upper_rr_threshold", plugin.params["upper_rr_threshold"]),
                int(loaded_params.get("time_horizon", 3))
            ]
            print(f"Evaluating strategy with loaded parameters: {candidate}")
            # Initialize optimizer globals so that evaluate_individual() can use them
            init_optimizer(plugin, base_data, hourly_preds, daily_preds, config)
            result = evaluate_individual(candidate)
            trading_info = {"best_parameters": {
                "profit_threshold": candidate[0],
                "tp_multiplier": candidate[1],
                "sl_multiplier": candidate[2],
                "lower_rr_threshold": candidate[3],
                "upper_rr_threshold": candidate[4],
                "time_horizon": candidate[5]
            }, "profit": result[0]}
        else:
            trading_info = {}
    else:
        # Otherwise, run full optimization.
        if hasattr(plugin, "get_optimizable_params") and hasattr(plugin, "evaluate_candidate"):
            print(f"\nPlugin supports optimization. Running optimizer for '{strat_name}'...")
            trading_info = run_optimizer(plugin, base_data, hourly_preds, daily_preds, config)
        else:
            print("\nPlugin does not support optimization. Exiting.")
            trading_info = {}

    print("\n=== Optimization Results ===")
    for key, value in trading_info.items():
        print(f"{key}: {value}")

    # Rename the final balance plot if 'balance_plot_file' is provided.
    if config.get("balance_plot_file"):
        old_plot = "balance_plot.png"
        new_plot = config["balance_plot_file"]
        if os.path.exists(old_plot):
            try:
                os.rename(old_plot, new_plot)
                print(f"Renamed {old_plot} -> {new_plot}")
            except Exception as e:
                print(f"Failed to rename {old_plot} to {new_plot}: {e}")
        else:
            print(f"Warning: {old_plot} not found; no balance plot to rename.")

    # Save trades to CSV if 'trades_csv_file' is specified.
    trades_csv = config.get("trades_csv_file")
    if trades_csv:
        try:
            if hasattr(plugin, "trades") and plugin.trades:
                pd.DataFrame(plugin.trades).to_csv(trades_csv, index=False)
                print(f"Trades saved to {trades_csv}.")
            else:
                print("Warning: plugin.trades not found or empty.")
        except Exception as e:
            print(f"Failed to save trades to {trades_csv}: {e}")

    # Save summary CSV if 'summary_csv_file' is specified.
    summary_csv = config.get("summary_csv_file")
    if summary_csv:
        try:
            df = pd.DataFrame([trading_info])
            df.to_csv(summary_csv, index=False)
            print(f"Summary saved to {summary_csv}.")
        except Exception as e:
            print(f"Failed to save summary CSV to {summary_csv}: {e}")

    # Save best parameters as JSON if in optimization mode and load_parameters is not provided.
    if config.get("load_parameters") is None and config.get("save_parameters"):
        try:
            with open(config["save_parameters"], "w") as f:
                json.dump(trading_info.get("best_parameters", {}), f, indent=4, default=str)
            print(f"Best parameters saved to {config['save_parameters']}.")
        except Exception as e:
            print(f"Failed to save best parameters to {config['save_parameters']}: {e}")

    end_time = time.time()
    print(f"\nTotal Execution Time: {end_time - start_time:.2f} seconds")
    return trading_info, getattr(plugin, "trades", None)


def run_processing_pipeline(config, plugin):
    """
    Executes the trading strategy optimization pipeline.
    
    - Loads and processes datasets.
    - If config["load_parameters"] is provided (not None), loads candidate parameters from the specified JSON file
      and evaluates the strategy once using those parameters (printing all trades, summary, and saving outputs).
    - Otherwise, runs the full optimization via run_optimizer().
    - At the end, renames the balance plot and saves the trades and summary CSV files.
    - If in optimization mode and if config["save_parameters"] is provided, saves the best parameters as JSON.
    """
    import json, os, pandas as pd
    from app.optimizer import init_optimizer, evaluate_individual, run_optimizer
    import time

    start_time = time.time()
    strat_name = config.get("strategy_name", "Heuristic Strategy")
    print(f"\n=== Starting Trading Strategy Optimization Pipeline for '{strat_name}' ===")

    datasets = process_data(config)
    hourly_preds, daily_preds, base_data = datasets["hourly"], datasets["daily"], datasets["base"]

    print("\nProcessed Dataset Shapes:")
    print(f"  Hourly predictions: {hourly_preds.shape}")
    print(f"  Daily predictions:  {daily_preds.shape}")
    print(f"  Base dataset:       {base_data.shape}")

    # If load_parameters is provided, load candidate parameters and evaluate the strategy once.
    if config.get("load_parameters") is not None:
        try:
            with open(config["load_parameters"], "r") as f:
                loaded_params = json.load(f)
            print(f"Loaded evaluation parameters from {config['load_parameters']}: {loaded_params}")
        except Exception as e:
            print(f"Failed to load parameters from {config['load_parameters']}: {e}")
            loaded_params = None
        if loaded_params is not None:
            # Construct candidate in the same order as get_optimizable_params():
            candidate = [
                loaded_params.get("profit_threshold", plugin.params["profit_threshold"]),
                loaded_params.get("tp_multiplier", plugin.params["tp_multiplier"]),
                loaded_params.get("sl_multiplier", plugin.params["sl_multiplier"]),
                loaded_params.get("lower_rr_threshold", plugin.params["lower_rr_threshold"]),
                loaded_params.get("upper_rr_threshold", plugin.params["upper_rr_threshold"]),
                int(loaded_params.get("time_horizon", 3))
            ]
            print(f"Evaluating strategy with loaded parameters: {candidate}")
            # Initialize optimizer globals so that evaluate_individual() can use them
            init_optimizer(plugin, base_data, hourly_preds, daily_preds, config)
            result = evaluate_individual(candidate)
            trading_info = {"best_parameters": {
                "profit_threshold": candidate[0],
                "tp_multiplier": candidate[1],
                "sl_multiplier": candidate[2],
                "lower_rr_threshold": candidate[3],
                "upper_rr_threshold": candidate[4],
                "time_horizon": candidate[5]
            }, "profit": result[0]}
        else:
            trading_info = {}
    else:
        # Otherwise, run full optimization.
        if hasattr(plugin, "get_optimizable_params") and hasattr(plugin, "evaluate_candidate"):
            print(f"\nPlugin supports optimization. Running optimizer for '{strat_name}'...")
            trading_info = run_optimizer(plugin, base_data, hourly_preds, daily_preds, config)
        else:
            print("\nPlugin does not support optimization. Exiting.")
            trading_info = {}

    print("\n=== Optimization Results ===")
    for key, value in trading_info.items():
        print(f"{key}: {value}")

    # Rename the final balance plot if 'balance_plot_file' is provided.
    if config.get("balance_plot_file"):
        old_plot = "balance_plot.png"
        new_plot = config["balance_plot_file"]
        if os.path.exists(old_plot):
            try:
                os.rename(old_plot, new_plot)
                print(f"Renamed {old_plot} -> {new_plot}")
            except Exception as e:
                print(f"Failed to rename {old_plot} to {new_plot}: {e}")
        else:
            print(f"Warning: {old_plot} not found; no balance plot to rename.")

    # Save trades to CSV if 'trades_csv_file' is specified.
    trades_csv = config.get("trades_csv_file")
    if trades_csv:
        try:
            if hasattr(plugin, "trades") and plugin.trades:
                pd.DataFrame(plugin.trades).to_csv(trades_csv, index=False)
                print(f"Trades saved to {trades_csv}.")
            else:
                print("Warning: plugin.trades not found or empty.")
        except Exception as e:
            print(f"Failed to save trades to {trades_csv}: {e}")

    # Save summary CSV if 'summary_csv_file' is specified.
    summary_csv = config.get("summary_csv_file")
    if summary_csv:
        try:
            df = pd.DataFrame([trading_info])
            df.to_csv(summary_csv, index=False)
            print(f"Summary saved to {summary_csv}.")
        except Exception as e:
            print(f"Failed to save summary CSV to {summary_csv}: {e}")

    # Save best parameters as JSON if in optimization mode and load_parameters is not provided.
    if config.get("load_parameters") is None and config.get("save_parameters"):
        try:
            with open(config["save_parameters"], "w") as f:
                json.dump(trading_info.get("best_parameters", {}), f, indent=4, default=str)
            print(f"Best parameters saved to {config['save_parameters']}.")
        except Exception as e:
            print(f"Failed to save best parameters to {config['save_parameters']}: {e}")

    end_time = time.time()
    print(f"\nTotal Execution Time: {end_time - start_time:.2f} seconds")
    return trading_info, getattr(plugin, "trades", None)



if __name__ == "__main__":
    pass
Para calcular el **MAE (Mean Absolute Error)** y el **R² (Coeficiente de determinación)** para cada predicción horaria y diaria respecto al valor de la columna 'CLOSE' del dataset base y mostrarlos en una tabla antes de pasar los datos al plugin, he realizado las siguientes modificaciones en tu código: 1. Se han importado las funciones necesarias para el cálculo de MAE y R² desde `sklearn.metrics`. 2. Se añadió un bloque de código en la función `process_data` para calcular y mostrar los resultados en una tabla con el MAE y R² para cada predicción horaria y diaria. 3. Se imprimirá esta tabla antes de enviar los datos al plugin. Aquí te dejo el código actualizado con estas modificaciones: ```python from sklearn.metrics import mean_absolute_error, r2_score def process_data(config): """ Loads and processes datasets, ensuring alignment and calculating MAE and R^2. - Uses external prediction files if provided. - Generates predictions if files are not available. - Ensures all datasets (base, hourly predictions, daily predictions) are properly aligned to the common date range. """ import pandas as pd import numpy as np import json from app.data_handler import load_csv headers = config.get("headers", True) print("Loading datasets...") # Load datasets based on config parameters hourly_df = load_csv(config["hourly_predictions_file"], headers=headers) if config.get("hourly_predictions_file") else None daily_df = load_csv(config["daily_predictions_file"], headers=headers) if config.get("daily_predictions_file") else None base_df = load_csv(config["base_dataset_file"], headers=headers) print(f"Base dataset loaded: {base_df.shape}") # Auto-generate predictions if files are missing if hourly_df is None: if "time_horizon" not in config or not config["time_horizon"]: raise ValueError("time_horizon must be provided when auto-generating predictions.") print("Auto-generating hourly predictions...") hourly_df = create_hourly_predictions(base_df, config["time_horizon"]) if daily_df is None: if "time_horizon" not in config or not config["time_horizon"]: raise ValueError("time_horizon must be provided when auto-generating predictions.") print("Auto-generating daily predictions...") daily_df = create_daily_predictions(base_df, config["time_horizon"]) # Ensure that hourly and daily predictions have a datetime index based on DATE_TIME column, if not already set. if hourly_df is not None: if not isinstance(hourly_df.index, pd.DatetimeIndex): if "DATE_TIME" in hourly_df.columns: hourly_df.index = pd.to_datetime(hourly_df["DATE_TIME"]) else: raise ValueError("hourly_df does not have a DATE_TIME column.") if daily_df is not None: if not isinstance(daily_df.index, pd.DatetimeIndex): if "DATE_TIME" in daily_df.columns: daily_df.index = pd.to_datetime(daily_df["DATE_TIME"]) else: raise ValueError("daily_df does not have a DATE_TIME column.") # Ensure base_df has a datetime index as well. if not isinstance(base_df.index, pd.DatetimeIndex): if "DATE_TIME" in base_df.columns: base_df.index = pd.to_datetime(base_df["DATE_TIME"]) else: raise ValueError("base_df does not have a DATE_TIME column.") # Align all datasets to the common date range (intersection of their datetime indexes) common_index = base_df.index.intersection(hourly_df.index).intersection(daily_df.index) if common_index.empty: raise ValueError("No common date range found among base, hourly, and daily predictions.") # Recortamos los tres datasets al rango común base_df = base_df.loc[common_index] # <-- Alinea el dataset base hourly_df = hourly_df.loc[common_index] daily_df = daily_df.loc[common_index] # Verify that all datasets have the same number of rows if not (len(base_df) == len(hourly_df) == len(daily_df)): raise ValueError("After alignment, the number of rows in base, hourly, and daily predictions do not match!") # Print aligned date ranges print(f"Aligned Base dataset range: {base_df.index.min()} to {base_df.index.max()}") print(f"Aligned Hourly predictions range: {hourly_df.index.min()} to {hourly_df.index.max()}") print(f"Aligned Daily predictions range: {daily_df.index.min()} to {daily_df.index.max()}") # Calculate MAE and R^2 for hourly predictions hourly_metrics = [] for i, column in enumerate(hourly_df.columns): mae = mean_absolute_error(base_df['CLOSE'], hourly_df[column]) r2 = r2_score(base_df['CLOSE'], hourly_df[column]) hourly_metrics.append([column, mae, r2]) # Calculate MAE and R^2 for daily predictions daily_metrics = [] for i, column in enumerate(daily_df.columns): mae = mean_absolute_error(base_df['CLOSE'], daily_df[column]) r2 = r2_score(base_df['CLOSE'], daily_df[column]) daily_metrics.append([column, mae, r2]) # Print the metrics in a table print("\nHourly Predictions - MAE and R^2:") print(pd.DataFrame(hourly_metrics, columns=['Hour', 'MAE', 'R^2'])) print("\nDaily Predictions - MAE and R^2:") print(pd.DataFrame(daily_metrics, columns=['Day', 'MAE', 'R^2'])) return {"hourly": hourly_df, "daily": daily_df, "base": base_df} ``` ### Resumen de los cambios realizados: 1. Calculamos el MAE y R² para cada columna de las predicciones horarias y diarias, comparándolas con la columna 'CLOSE' del dataset base. 2. Los resultados se imprimen en una tabla para cada conjunto de predicciones (horarias y diarias) antes de pasar los datos al plugin de optimización. Con esta actualización, cuando ejecutas el script, verás las métricas de MAE y R² para las predicciones antes de que los datos sean procesados por el plugin de estrategia. durante 42 segundos
Aquí te presento el código actualizado de los métodos afectados. Se ha modificado la función process_data para conservar una copia completa del dataset base (sin recortar) en la variable base_full. Luego, en run_processing_pipeline se calcula, para cada horizonte de predicción (en horas y en días), el MAE y el R² usando como target el valor de la columna CLOSE obtenido del dataset base completo, tomando el tiempo de emisión + el desfase correspondiente. Finalmente se imprime una tabla para cada modalidad antes de pasar los datos al plugin.

Observa que la alineación de los tres datasets (base, predicciones horarias y predicciones diarias) se realiza como antes y la línea que alinea el dataset base es:

base_df = base_df.loc[common_index]
La versión completa actualizada es la siguiente:

def process_data(config):
    """
    Loads and processes datasets, ensuring alignment and applying max_steps.
    - Uses external prediction files if provided.
    - Generates predictions if files are not available.
    - Ensures all datasets (base, hourly predictions, daily predictions) are properly aligned to the common date range.
      Additionally, preserves the full base dataset (base_full) for later evaluation.
    """
    import pandas as pd
    import numpy as np
    import json
    from app.data_handler import load_csv

    headers = config.get("headers", True)
    print("Loading datasets...")

    # Load datasets based on config parameters
    hourly_df = load_csv(config["hourly_predictions_file"], headers=headers) if config.get("hourly_predictions_file") else None
    daily_df = load_csv(config["daily_predictions_file"], headers=headers) if config.get("daily_predictions_file") else None
    base_df = load_csv(config["base_dataset_file"], headers=headers)
    print(f"Base dataset loaded: {base_df.shape}")

    # Conserva una copia completa del base dataset para la evaluación
    base_df_full = base_df.copy()

    # Auto-generate predictions if files are missing
    if hourly_df is None:
        if "time_horizon" not in config or not config["time_horizon"]:
            raise ValueError("time_horizon must be provided when auto-generating predictions.")
        print("Auto-generating hourly predictions...")
        hourly_df = create_hourly_predictions(base_df, config["time_horizon"])

    if daily_df is None:
        if "time_horizon" not in config or not config["time_horizon"]:
            raise ValueError("time_horizon must be provided when auto-generating predictions.")
        print("Auto-generating daily predictions...")
        daily_df = create_daily_predictions(base_df, config["time_horizon"])

    # Ensure that hourly and daily predictions have a datetime index based on DATE_TIME column, if not already set.
    if hourly_df is not None:
        if not isinstance(hourly_df.index, pd.DatetimeIndex):
            if "DATE_TIME" in hourly_df.columns:
                hourly_df.index = pd.to_datetime(hourly_df["DATE_TIME"])
            else:
                raise ValueError("hourly_df does not have a DATE_TIME column.")
    if daily_df is not None:
        if not isinstance(daily_df.index, pd.DatetimeIndex):
            if "DATE_TIME" in daily_df.columns:
                daily_df.index = pd.to_datetime(daily_df["DATE_TIME"])
            else:
                raise ValueError("daily_df does not have a DATE_TIME column.")

    # Ensure base_df has a datetime index as well.
    if not isinstance(base_df.index, pd.DatetimeIndex):
        if "DATE_TIME" in base_df.columns:
            base_df.index = pd.to_datetime(base_df["DATE_TIME"])
        else:
            raise ValueError("base_df does not have a DATE_TIME column.")

    # Align all datasets to the common date range (intersection of their datetime indexes)
    common_index = base_df.index.intersection(hourly_df.index).intersection(daily_df.index)
    if common_index.empty:
        raise ValueError("No common date range found among base, hourly, and daily predictions.")

    # Aquí se recortan los tres datasets al rango común
    base_df = base_df.loc[common_index]         # <-- Alinea el dataset base
    hourly_df = hourly_df.loc[common_index]
    daily_df = daily_df.loc[common_index]

    # Verify that all datasets have the same number of rows
    if not (len(base_df) == len(hourly_df) == len(daily_df)):
        raise ValueError("After alignment, the number of rows in base, hourly, and daily predictions do not match!")

    # Print aligned date ranges
    print(f"Aligned Base dataset range: {base_df.index.min()} to {base_df.index.max()}")
    print(f"Aligned Hourly predictions range: {hourly_df.index.min()} to {hourly_df.index.max()}")
    print(f"Aligned Daily predictions range: {daily_df.index.min()} to {daily_df.index.max()}")

    return {"hourly": hourly_df, "daily": daily_df, "base": base_df, "base_full": base_df_full}


def run_processing_pipeline(config, plugin):
    """
    Executes the trading strategy optimization pipeline.
    
    - Loads and processes datasets.
    - Before sending data to the strategy plugin, computes and prints a table with MAE and R² for each prediction horizon,
      using as target el valor de la columna CLOSE del dataset base (full).
    - If config["load_parameters"] is provided, loads candidate parameters from the specified JSON file and evaluates the strategy once.
    - Otherwise, runs the full optimization via run_optimizer().
    - At the end, renames the balance plot and saves the trades and summary CSV files.
    - If in optimization mode and config["save_parameters"] is provided, saves the best parameters as JSON.
    """
    import json, os, pandas as pd, time
    from app.optimizer import init_optimizer, evaluate_individual, run_optimizer
    from sklearn.metrics import mean_absolute_error, r2_score

    start_time = time.time()
    strat_name = config.get("strategy_name", "Heuristic Strategy")
    print(f"\n=== Starting Trading Strategy Optimization Pipeline for '{strat_name}' ===")

    datasets = process_data(config)
    hourly_preds = datasets["hourly"]
    daily_preds = datasets["daily"]
    base_data = datasets["base"]         # Aligned base (issuance times)
    base_full = datasets["base_full"]      # Base dataset completo

    # Calcular métricas de error para cada horizonte de predicción

    # Para predicciones horarias: cada columna corresponde a la predicción para h horas en el futuro.
    n_hourly = hourly_preds.shape[1]
    hourly_results = []
    for h in range(1, n_hourly+1):
        # Para cada tiempo de emisión, el target real es el valor de CLOSE en (tiempo + h horas)
        forecast_times = hourly_preds.index + pd.Timedelta(hours=h)
        actual = base_full.reindex(forecast_times)["CLOSE"]
        pred = hourly_preds.iloc[:, h-1]
        # Solo evaluamos donde tengamos valor real
        valid = actual.notna()
        if valid.sum() == 0:
            mae = None
            r2 = None
        else:
            mae = mean_absolute_error(actual[valid], pred[valid])
            r2 = r2_score(actual[valid], pred[valid])
        hourly_results.append({"Horizon (hours)": h, "MAE": mae, "R2": r2})
    
    df_hourly = pd.DataFrame(hourly_results)

    # Para predicciones diarias: cada columna corresponde a la predicción para d días en el futuro (24*d horas)
    n_daily = daily_preds.shape[1]
    daily_results = []
    for d in range(1, n_daily+1):
        forecast_times = daily_preds.index + pd.Timedelta(hours=24*d)
        actual = base_full.reindex(forecast_times)["CLOSE"]
        pred = daily_preds.iloc[:, d-1]
        valid = actual.notna()
        if valid.sum() == 0:
            mae = None
            r2 = None
        else:
            mae = mean_absolute_error(actual[valid], pred[valid])
            r2 = r2_score(actual[valid], pred[valid])
        daily_results.append({"Horizon (days)": d, "MAE": mae, "R2": r2})
    
    df_daily = pd.DataFrame(daily_results)

    # Imprimir las tablas de métricas
    print("\nError Metrics for Hourly Predictions:")
    print(df_hourly.to_string(index=False))
    print("\nError Metrics for Daily Predictions:")
    print(df_daily.to_string(index=False))

    # Verificación adicional: imprimir rangos finales antes de enviar al plugin
    print(f"\nFinal Base dataset date range: {base_data.index.min()} to {base_data.index.max()}")
    print(f"Final Hourly predictions date range: {hourly_preds.index.min()} to {hourly_preds.index.max()}")
    print(f"Final Daily predictions date range: {daily_preds.index.min()} to {daily_preds.index.max()}")

    print("\nProcessed Dataset Shapes:")
    print(f"  Base dataset:       {base_data.shape}")
    print(f"  Hourly predictions: {hourly_preds.shape}")
    print(f"  Daily predictions:  {daily_preds.shape}")

    # A continuación, se envían los datos al plugin (evaluación o optimización)
    if config.get("load_parameters") is not None:
        try:
            with open(config["load_parameters"], "r") as f:
                loaded_params = json.load(f)
            print(f"Loaded evaluation parameters from {config['load_parameters']}: {loaded_params}")
        except Exception as e:
            print(f"Failed to load parameters from {config['load_parameters']}: {e}")
            loaded_params = None
        if loaded_params is not None:
            candidate = [
                loaded_params.get("profit_threshold", plugin.params["profit_threshold"]),
                loaded_params.get("tp_multiplier", plugin.params["tp_multiplier"]),
                loaded_params.get("sl_multiplier", plugin.params["sl_multiplier"]),
                loaded_params.get("lower_rr_threshold", plugin.params["lower_rr_threshold"]),
                loaded_params.get("upper_rr_threshold", plugin.params["upper_rr_threshold"]),
                int(loaded_params.get("time_horizon", 3))
            ]
            print(f"Evaluating strategy with loaded parameters: {candidate}")
            init_optimizer(plugin, base_data, hourly_preds, daily_preds, config)
            result = evaluate_individual(candidate)
            trading_info = {"best_parameters": {
                "profit_threshold": candidate[0],
                "tp_multiplier": candidate[1],
                "sl_multiplier": candidate[2],
                "lower_rr_threshold": candidate[3],
                "upper_rr_threshold": candidate[4],
                "time_horizon": candidate[5]
            }, "profit": result[0]}
        else:
            trading_info = {}
    else:
        if hasattr(plugin, "get_optimizable_params") and hasattr(plugin, "evaluate_candidate"):
            print(f"\nPlugin supports optimization. Running optimizer for '{strat_name}'...")
            trading_info = run_optimizer(plugin, base_data, hourly_preds, daily_preds, config)
        else:
            print("\nPlugin does not support optimization. Exiting.")
            trading_info = {}

    print("\n=== Optimization Results ===")
    for key, value in trading_info.items():
        print(f"{key}: {value}")

    if config.get("balance_plot_file"):
        old_plot = "balance_plot.png"
        new_plot = config["balance_plot_file"]
        if os.path.exists(old_plot):
            try:
                os.rename(old_plot, new_plot)
                print(f"Renamed {old_plot} -> {new_plot}")
            except Exception as e:
                print(f"Failed to rename {old_plot} to {new_plot}: {e}")
        else:
            print(f"Warning: {old_plot} not found; no balance plot to rename.")

    trades_csv = config.get("trades_csv_file")
    if trades_csv:
        try:
            if hasattr(plugin, "trades") and plugin.trades:
                pd.DataFrame(plugin.trades).to_csv(trades_csv, index=False)
                print(f"Trades saved to {trades_csv}.")
            else:
                print("Warning: plugin.trades not found or empty.")
        except Exception as e:
            print(f"Failed to save trades to {trades_csv}: {e}")

    summary_csv = config.get("summary_csv_file")
    if summary_csv:
        try:
            df = pd.DataFrame([trading_info])
            df.to_csv(summary_csv, index=False)
            print(f"Summary saved to {summary_csv}.")
        except Exception as e:
            print(f"Failed to save summary CSV to {summary_csv}: {e}")

    if config.get("load_parameters") is None and config.get("save_parameters"):
        try:
            with open(config["save_parameters"], "w") as f:
                json.dump(trading_info.get("best_parameters", {}), f, indent=4, default=str)
            print(f"Best parameters saved to {config['save_parameters']}.")
        except Exception as e:
            print(f"Failed to save best parameters to {config['save_parameters']}: {e}")

    end_time = time.time()
    print(f"\nTotal Execution Time: {end_time - start_time:.2f} seconds")
    return trading_info, getattr(plugin, "trades", None)

if __name__ == "__main__":
    pass
