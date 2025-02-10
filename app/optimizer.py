import backtrader as bt
import random
import datetime
from deap import base, creator, tools
import time
from tqdm import tqdm
import multiprocessing
import json

# Global variables for optimization
_plugin = None
_base_data = None
_hourly_predictions = None
_daily_predictions = None
_config = None
_current_epoch = 1  # Global variable to hold current epoch number

def init_optimizer(plugin, base_data, hourly_predictions, daily_predictions, config):
    """
    Initializes the optimizer with the provided plugin and datasets.
    """
    global _plugin, _base_data, _hourly_predictions, _daily_predictions, _config, _num_generations
    _plugin = plugin
    _base_data = base_data
    _hourly_predictions = hourly_predictions
    _daily_predictions = daily_predictions
    _config = config
    _num_generations = config.get("num_generations", 10)
    print("[INIT] Optimizer initialized with strategy plugin.")


def evaluate_individual(individual):
    """
    Evaluates a candidate strategy parameter set.
    Prints the current epoch number along with the candidate.
    Expects the plugin's evaluate_candidate() method to return either:
         - A tuple: (profit, stats) where stats is a dict containing keys 'num_trades', 'win_pct', 'max_dd', 'sharpe'
         - Or a single-value tuple (profit,)
    This function returns a tuple containing only the profit (of length 1), 
    so that it matches the fitness weight configuration.
    """
    global _plugin, _base_data, _hourly_predictions, _daily_predictions, _config, _current_epoch, _num_generations
    if _plugin is None:
        print("[EVALUATE] ERROR: _plugin is None!")
        return (-1e6,)
    
    # Print the candidate and current epoch information.
    print(f"[EVALUATE][Epoch {_current_epoch}/{_num_generations}] Evaluating candidate (genome): {individual}")
    
    result = _plugin.evaluate_candidate(individual, _base_data, _hourly_predictions, _daily_predictions, _config)
    
    # If the result returns both profit and stats, extract and print them.
    if isinstance(result, tuple) and len(result) == 2:
        profit, stats = result
        print(f"[EVALUATE][Epoch {_current_epoch}/{_num_generations}] Candidate result => Profit: {profit:.2f}, "
              f"Trades: {stats.get('num_trades', 0)}, "
              f"Win%: {stats.get('win_pct', 0):.1f}, "
              f"MaxDD: {stats.get('max_dd', 0):.2f}, "
              f"Sharpe: {stats.get('sharpe', 0):.2f}")
        return (profit,)
    # If only profit is returned as a single-value tuple, print and return that.
    elif isinstance(result, tuple) and len(result) == 1:
        print(f"[EVALUATE][Epoch {_current_epoch}/{_num_generations}] Candidate result => Profit: {result[0]:.2f} (no stats)")
        return result
    else:
        # Fallback: assume result is a single numeric value.
        print(f"[EVALUATE][Epoch {_current_epoch}/{_num_generations}] Candidate result => Profit: {result:.2f} (no stats)")
        return (result,)


def run_optimizer(plugin, base_data, hourly_predictions, daily_predictions, config):
    """
    Runs the optimizer using DEAP to optimize the strategy parameters.
    Displays a TQDM progress bar for the evaluation of individuals in each generation.
    Prints detailed candidate evaluation information including the current epoch.
    Saves the best found parameters as JSON if config['save_config'] is provided.
    """
    global _current_epoch
    from tqdm import tqdm
    import random
    import multiprocessing
    import json
    from deap import base, creator, tools

    # Initialize globals
    init_optimizer(plugin, base_data, hourly_predictions, daily_predictions, config)

    optimizable_params = plugin.get_optimizable_params()
    num_params = len(optimizable_params)
    print(f"Optimizable Parameters ({num_params}):")
    for name, low, high in optimizable_params:
        print(f"  {name}: [{low}, {high}]")

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

    population = toolbox.population(n=population_size)
    print("[OPTIMIZATION] Evaluating initial population...")
    if disable_mp:
        fitnesses = []
        with tqdm(total=len(population), desc="Initial eval", unit="cand") as pbar:
            for ind in population:
                fit = toolbox.evaluate(ind)
                ind.fitness.values = fit
                fitnesses.append(fit)
                pbar.update(1)
    else:
        fitnesses = []
        with tqdm(total=len(population), desc="Initial eval", unit="cand") as pbar:
            for fit in toolbox.map(toolbox.evaluate, population):
                fitnesses.append(fit)
                pbar.update(1)
        for ind, f in zip(population, fitnesses):
            ind.fitness.values = f

    print(f"  Evaluated {len(population)} individuals initially.")

    for gen in range(1, num_generations + 1):
        _current_epoch = gen+1
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

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        if disable_mp:
            with tqdm(total=len(invalid_ind), desc=f"Epoch {gen} eval", unit="cand") as pbar:
                for ind in invalid_ind:
                    fit = toolbox.evaluate(ind)
                    ind.fitness.values = fit
                    pbar.update(1)
        else:
            fitnesses = []
            with tqdm(total=len(invalid_ind), desc=f"Epoch {gen} eval", unit="cand") as pbar:
                for fit in toolbox.map(toolbox.evaluate, invalid_ind):
                    fitnesses.append(fit)
                    pbar.update(1)
            for ind, f in zip(invalid_ind, fitnesses):
                ind.fitness.values = f

        population[:] = offspring
        fits = [ind.fitness.values[0] for ind in population]
        print(f"Generation {gen}: Max Profit = {max(fits):.2f}, Avg Profit = {sum(fits) / len(fits):.2f}")

    from deap import tools
    best_ind = tools.selBest(population, 1)[0]
    print("[OPTIMIZATION] Best parameter set found:")
    best_params = {name: best_ind[i] for i, (name, _, _) in enumerate(optimizable_params)}
    for name, value in best_params.items():
        print(f"  {name} = {value:.4f}")
    print(f"Achieved Profit: {best_ind.fitness.values[0]:.2f}")

    if not disable_mp:
        pool.close()

    # Save the best parameters as JSON if configured (using config["save_config"] for best parameters).
    if config.get("save_config"):
        try:
            with open(config["save_config"], "w") as f:
                json.dump(best_params, f, indent=4, default=str)
            print(f"Best parameters saved to {config['save_config']}.")
        except Exception as e:
            print(f"Failed to save best parameters to {config['save_config']}: {e}")

    return {
        "best_parameters": best_params,
        "profit": best_ind.fitness.values[0],
    }

if __name__ == '__main__':
    print("Standalone testing of optimizer not supported; run via main pipeline.")


if __name__ == '__main__':
    print("Standalone testing of optimizer not supported; run via main pipeline.")
