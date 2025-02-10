import backtrader as bt
import pandas as pd
import datetime
import random
import time
import multiprocessing
import matplotlib.pyplot as plt

from deap import base, creator, tools

# Import your strategy.
# (Ensure that HeuristicStrategy is defined exactly as in your example.)
from heuristic_strategy import HeuristicStrategy

# --------------------------------------------------------------------------
# Global variables to hold the evaluation context.
# These will be set in run_optimizer and used in evaluate_individual.
# --------------------------------------------------------------------------
_plugin = None
_base_data = None
_hourly_predictions = None
_daily_predictions = None
_config = None

def evaluate_individual(individual):
    """
    Global evaluation function that calls the plugin's evaluate_candidate method.
    
    Args:
        individual (list): A candidate parameter set.
    
    Returns:
        tuple: A one-tuple containing the profit (fitness value).
    """
    global _plugin, _base_data, _hourly_predictions, _daily_predictions, _config
    return _plugin.evaluate_candidate(individual, _base_data, _hourly_predictions, _daily_predictions, _config)

def run_optimizer(plugin, base_data, hourly_predictions, daily_predictions, config):
    """
    Runs the DEAP-based optimizer using the strategy plugin's evaluation function.
    
    The plugin must implement two methods:
      - get_optimizable_params(): returns a list of tuples (name, lower_bound, upper_bound).
      - evaluate_candidate(individual, base_data, hourly_predictions, daily_predictions, config):
            runs a backtest with the candidate parameters and returns its profit as a tuple.
    
    Args:
        plugin: The strategy plugin instance.
        base_data (pd.DataFrame): The base dataset with actual rates.
        hourly_predictions (pd.DataFrame): Hourly predictions.
        daily_predictions (pd.DataFrame): Daily predictions.
        config (dict): The configuration dictionary.
    
    Returns:
        dict: A dictionary containing "best_parameters" (the best candidate's parameters) and "profit" (the best profit achieved).
    """
    global _plugin, _base_data, _hourly_predictions, _daily_predictions, _config
    _plugin = plugin
    _base_data = base_data
    _hourly_predictions = hourly_predictions
    _daily_predictions = daily_predictions
    _config = config

    # Retrieve optimizable parameters from the plugin.
    optimizable_params = plugin.get_optimizable_params()
    num_params = len(optimizable_params)
    print(f"Optimizable Parameters ({num_params}):")
    for name, low, high in optimizable_params:
         print(f"  {name}: [{low}, {high}]")
    
    # Create DEAP classes if not already defined.
    if not hasattr(creator, "FitnessMax"):
         creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
         creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    
    # Register an individual generator that creates a list of random values within the specified ranges.
    def random_attr(param):
         name, low, high = param
         return random.uniform(low, high)
    toolbox.register("individual", lambda: creator.Individual([random_attr(param) for param in optimizable_params]))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Register the global evaluation function.
    toolbox.register("evaluate", evaluate_individual)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1.0, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    random.seed(42)
    population_size = config.get("population_size", 20)
    num_generations = config.get("num_generations", 100)
    cxpb = config.get("crossover_probability", 0.5)
    mutpb = config.get("mutation_probability", 0.2)
    
    population = toolbox.population(n=population_size)
    print("Starting Genetic Algorithm Optimization...")
    print(f"Population Size: {population_size}, Generations: {num_generations}")
    
    # Use multiprocessing to speed up evaluation.
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)
    
    # Evaluate the initial population.
    fitnesses = list(toolbox.map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
         ind.fitness.values = fit
    print(f"  Evaluated {len(population)} individuals initially.")
    
    # Evolution loop.
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
         fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
         for ind, fit in zip(invalid_ind, fitnesses):
              ind.fitness.values = fit
         
         population[:] = offspring
         fits = [ind.fitness.values[0] for ind in population]
         print(f"Generation {gen}: Max Profit = {max(fits):.2f}, Avg Profit = {sum(fits)/len(fits):.2f}")
    
    best_ind = tools.selBest(population, 1)[0]
    print("Best parameter set found:")
    best_params = {}
    for i, (name, low, high) in enumerate(optimizable_params):
         best_params[name] = best_ind[i]
         print(f"  {name} = {best_ind[i]:.4f}")
    print(f"Achieved Profit: {best_ind.fitness.values[0]:.2f}")
    
    pool.close()
    
    return {"best_parameters": best_params, "profit": best_ind.fitness.values[0]}

# Standalone testing of the optimizer is possible by calling main() here.
def main():
    random.seed(42)
    population = toolbox.population(n=20)
    CXPB, MUTPB, NGEN = 0.5, 0.2, 100  # For standalone testing.
    print("Starting Genetic Algorithm Optimization")
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
         ind.fitness.values = fit
    print("  Evaluated %i individuals" % len(population))
    for gen in range(1, NGEN + 1):
         offspring = toolbox.select(population, len(population))
         offspring = list(map(toolbox.clone, offspring))
         for child1, child2 in zip(offspring[::2], offspring[1::2]):
              if random.random() < CXPB:
                   toolbox.mate(child1, child2)
                   del child1.fitness.values
                   del child2.fitness.values
         for mutant in offspring:
              if random.random() < MUTPB:
                   toolbox.mutate(mutant)
                   del mutant.fitness.values
         invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
         fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
         for ind, fit in zip(invalid_ind, fitnesses):
              ind.fitness.values = fit
         population[:] = offspring
         fits = [ind.fitness.values[0] for ind in population]
         print(f"Generation {gen}: Max Profit = {max(fits):.2f}, Avg Profit = {sum(fits)/len(fits):.2f}")
    best_ind = tools.selBest(population, 1)[0]
    print("Best parameter set found:")
    print(f"  profit_threshold = {best_ind[0]:.2f}")
    print(f"  tp_multiplier    = {best_ind[1]:.2f}")
    print(f"  sl_multiplier    = {best_ind[2]:.2f}")
    print(f"  rel_volume       = {best_ind[3]:.3f}")
    print(f"  lower_rr_thresh  = {best_ind[4]:.2f}")
    print(f"  upper_rr_thresh  = {best_ind[5]:.2f}")
    print("With profit: {:.2f}".format(best_ind.fitness.values[0]))
    pool.close()

if __name__ == '__main__':
    main()
