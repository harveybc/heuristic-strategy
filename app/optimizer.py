import backtrader as bt
import random
import datetime
from deap import base, creator, tools
import time

# Global variables for optimization
_plugin = None
_base_data = None
_hourly_predictions = None
_daily_predictions = None
_config = None

def init_optimizer(plugin, base_data, hourly_predictions, daily_predictions, config):
    """
    Initializes the optimizer with the provided plugin and datasets.
    """
    global _plugin, _base_data, _hourly_predictions, _daily_predictions, _config
    _plugin = plugin
    _base_data = base_data
    _hourly_predictions = hourly_predictions
    _daily_predictions = daily_predictions
    _config = config
    print("[INIT] Optimizer initialized with strategy plugin.")

def evaluate_individual(individual):
    """
    Evaluates a candidate strategy parameter set, printing
    final balance, # of trades, trades won %, max dd, sharpe ratio, etc.
    as returned by the plugin.
    """
    global _plugin, _base_data, _hourly_predictions, _daily_predictions, _config
    if _plugin is None:
        print("[EVALUATE] ERROR: _plugin is None!")
        return (-1e6,)

    # Print the candidate
    print(f"[EVALUATE] Evaluating candidate (genome): {individual}")

    # We assume evaluate_candidate returns something like:
    # (profit_value, {"num_trades": ..., "win_pct": ..., "max_dd": ..., "sharpe": ...})
    result = _plugin.evaluate_candidate(individual, _base_data, _hourly_predictions, _daily_predictions, _config)

    if isinstance(result, tuple) and len(result) == 2:
        # We expect first item => profit, second => stats
        profit = result[0]
        stats = result[1]
        # Print the stats we want:
        print(f"[EVALUATE] Candidate result => Profit: {profit:.2f}, "
              f"Trades: {stats.get('num_trades', 0)}, "
              f"Win%: {stats.get('win_pct', 0):.1f}, "
              f"MaxDD: {stats.get('max_dd', 0):.2f}, "
              f"Sharpe: {stats.get('sharpe', 0):.2f}")
        # Return the profit alone as the fitness for DEAP
        return (profit,)

    # If the plugin just returns a single float, handle gracefully
    elif isinstance(result, tuple) and len(result) == 1:
        print(f"[EVALUATE] Candidate result => Profit: {result[0]:.2f} (no stats)")
        return result
    else:
        # fallback if plugin returns just a float
        print(f"[EVALUATE] Candidate result => Profit: {result:.2f} (no stats)")
        return (result,)


from tqdm import tqdm

from tqdm import tqdm

def run_optimizer(plugin, base_data, hourly_predictions, daily_predictions, config):
    """
    Runs the optimizer using DEAP to optimize the strategy parameters,
    displaying a TQDM progress bar for the evaluation of individuals
    in each generation (including the initial population).
    Prints the same final lines as the stand-alone code.
    """
    import random
    import multiprocessing
    from deap import base, creator, tools

    # Initialize
    global _plugin, _base_data, _hourly_predictions, _daily_predictions, _config
    _plugin = plugin
    _base_data = base_data
    _hourly_predictions = hourly_predictions
    _daily_predictions = daily_predictions
    _config = config

    optimizable_params = plugin.get_optimizable_params()

    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    def random_attr(param):
        name, low, high = param
        return random.uniform(low, high)

    toolbox.register("individual", lambda: creator.Individual([random_attr(p) for p in optimizable_params]))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate_individual)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1.0, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    random.seed(42)
    population_size = config.get("population_size", 20)
    num_generations = config.get("num_generations", 50)
    cxpb = config.get("crossover_probability", 0.5)
    mutpb = config.get("mutation_probability", 0.2)

    print("Starting Genetic Algorithm Optimization")

    disable_mp = config.get("disable_multiprocessing", False)
    if not disable_mp:
        pool = multiprocessing.Pool()
        toolbox.register("map", pool.map)

    # 1) Build the initial population
    population = toolbox.population(n=population_size)

    # 2) Evaluate the initial population with a TQDM progress bar
    if disable_mp:
        fitnesses = []
        with tqdm(total=len(population), desc="Evaluating initial population", unit="cand") as pbar:
            for ind in population:
                fit = toolbox.evaluate(ind)
                ind.fitness.values = fit
                pbar.update(1)
    else:
        # In parallel, we'll create a list out of the map but still track progress
        # Easiest is to do a small helper in a loop:
        fitnesses = []
        with tqdm(total=len(population), desc="Evaluating initial population", unit="cand") as pbar:
            for fit in toolbox.map(toolbox.evaluate, population):
                fitnesses.append(fit)
                pbar.update(1)
        # Assign results
        for ind, f in zip(population, fitnesses):
            ind.fitness.values = f

    print(f"  Evaluated {len(population)} individuals")

    # 3) The evolution loop
    for gen in range(1, num_generations + 1):
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate new individuals (invalid_ind)
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

        if disable_mp:
            with tqdm(total=len(invalid_ind), desc=f"Gen {gen} evaluating", unit="cand") as pbar:
                for ind in invalid_ind:
                    fit = toolbox.evaluate(ind)
                    ind.fitness.values = fit
                    pbar.update(1)
        else:
            # In parallel
            fitnesses = []
            with tqdm(total=len(invalid_ind), desc=f"Gen {gen} evaluating", unit="cand") as pbar:
                for fit in toolbox.map(toolbox.evaluate, invalid_ind):
                    fitnesses.append(fit)
                    pbar.update(1)
            for ind, f in zip(invalid_ind, fitnesses):
                ind.fitness.values = f

        population[:] = offspring
        fits = [ind.fitness.values[0] for ind in population]
        print(f"Generation {gen}: Max Profit = {max(fits):.2f}, Avg Profit = {sum(fits)/len(fits):.2f}")

    # 4) Retrieve best
    from deap import tools
    best_ind = tools.selBest(population, 1)[0]
    print("Best parameter set found:")
    print(f"  profit_threshold = {best_ind[0]:.2f}")
    print(f"  tp_multiplier    = {best_ind[1]:.2f}")
    print(f"  sl_multiplier    = {best_ind[2]:.2f}")
    print(f"  rel_volume       = {best_ind[3]:.3f}")
    print(f"  lower_rr_thresh  = {best_ind[4]:.2f}")
    print(f"  upper_rr_thresh  = {best_ind[5]:.2f}")
    print("With profit: {:.2f}".format(best_ind.fitness.values[0]))

    if not disable_mp:
        pool.close()

    return {
        "best_parameters": {
            "profit_threshold": best_ind[0],
            "tp_multiplier": best_ind[1],
            "sl_multiplier": best_ind[2],
            "rel_volume": best_ind[3],
            "lower_rr_threshold": best_ind[4],
            "upper_rr_threshold": best_ind[5],
        },
        "profit": best_ind.fitness.values[0],
    }



