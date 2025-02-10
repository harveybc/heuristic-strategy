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
    """
    init_optimizer(plugin, base_data, hourly_predictions, daily_predictions, config)

    # Get optimizable parameters from the plugin
    optimizable_params = plugin.get_optimizable_params()
    num_params = len(optimizable_params)
    print(f"Optimizable Parameters ({num_params}):")
    for name, low, high in optimizable_params:
        print(f"  {name}: [{low}, {high}]")

    # DEAP setup
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Register parameter generators
    def random_attr(param):
        name, low, high = param
        return random.uniform(low, high)

    toolbox.register("individual", lambda: creator.Individual([random_attr(param) for param in optimizable_params]))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Genetic Algorithm setup
    toolbox.register("evaluate", evaluate_individual)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1.0, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    random.seed(42)
    population_size = config.get("population_size", 20)
    num_generations = config.get("num_generations", 50)
    cxpb = config.get("crossover_probability", 0.5)
    mutpb = config.get("mutation_probability", 0.2)

    population = toolbox.population(n=population_size)
    print("[OPTIMIZATION] Starting Genetic Algorithm...")
    print(f"Population Size: {population_size}, Generations: {num_generations}")

    # Evaluate the initial population
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit
    print(f"  Evaluated {len(population)} individuals initially.")

    # Evolution loop
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

        # Evaluate new population
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        population[:] = offspring
        fits = [ind.fitness.values[0] for ind in population]
        print(f"Generation {gen}: Max Profit = {max(fits):.2f}, Avg Profit = {sum(fits) / len(fits):.2f}")

    # Retrieve the best individual
    best_ind = tools.selBest(population, 1)[0]
    print("[OPTIMIZATION] Best parameter set found:")
    best_params = {name: best_ind[i] for i, (name, _, _) in enumerate(optimizable_params)}
    for name, value in best_params.items():
        print(f"  {name} = {value:.4f}")
    print(f"Achieved Profit: {best_ind.fitness.values[0]:.2f}")

    return {"best_parameters": best_params, "profit": best_ind.fitness.values[0]}
