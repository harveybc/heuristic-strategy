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
    Evaluates a candidate strategy parameter set.
    """
    global _plugin, _base_data, _hourly_predictions, _daily_predictions, _config
    if _plugin is None:
        print("[EVALUATE] ERROR: _plugin is None!")
        return (-1e6,)

    print(f"[EVALUATE] Evaluating candidate: {individual}")
    result = _plugin.evaluate_candidate(individual, _base_data, _hourly_predictions, _daily_predictions, _config)
    print(f"[EVALUATE] Candidate result: {result}")
    return result

def run_optimizer(plugin, base_data, hourly_predictions, daily_predictions, config):
    """
    Runs the optimizer using DEAP to optimize the strategy parameters.
    Respects 'disable_multiprocessing' to decide whether to spawn processes or not.
    Prints the same messages as the stand-alone code.
    """
    from deap import base, creator, tools
    import random
    import time
    import multiprocessing

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

    toolbox.register("individual", lambda: creator.Individual([random_attr(param) for param in optimizable_params]))
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

    # Decide whether to spawn a Pool
    disable_mp = config.get("disable_multiprocessing", False)
    if not disable_mp:
        pool = multiprocessing.Pool()
        toolbox.register("map", pool.map)
    # else we rely on built-in python map => single process

    # Build population & evaluate
    population = toolbox.population(n=population_size)
    if disable_mp:
        fitnesses = list(map(toolbox.evaluate, population))
    else:
        fitnesses = list(toolbox.map(toolbox.evaluate, population))

    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit
    print("  Evaluated %i individuals" % len(population))

    # Evolve
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

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        if disable_mp:
            fitnesses = list(map(toolbox.evaluate, invalid_ind))
        else:
            fitnesses = list(toolbox.map(toolbox.evaluate, invalid_ind))

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        population[:] = offspring
        fits = [ind.fitness.values[0] for ind in population]
        print(f"Generation {gen}: Max Profit = {max(fits):.2f}, Avg Profit = {sum(fits)/len(fits):.2f}")

    # Best
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


