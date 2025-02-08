import random
import time
import multiprocessing
from deap import base, creator, tools

def run_optimizer(strategy_plugin, base_data, hourly_predictions, daily_predictions, config):
    """
    Optimize a trading strategy using a DEAP-based genetic algorithm.
    
    The strategy_plugin must implement:
      - get_optimizable_params(): returns a list of tuples (name, lower_bound, upper_bound).
      - evaluate_candidate(individual, base_data, hourly_predictions, daily_predictions, config):
            evaluates a candidate solution and returns a tuple with its fitness (profit).
    
    Args:
        strategy_plugin: Plugin instance implementing the optimization interface.
        base_data (pd.DataFrame): Actual rate data.
        hourly_predictions (pd.DataFrame): Hourly predictions data.
        daily_predictions (pd.DataFrame): Daily predictions data.
        config (dict): Optimizer configuration. Optional keys include:
            - population_size (default: 20)
            - num_generations (default: 100)
            - crossover_probability (default: 0.5)
            - mutation_probability (default: 0.2)
    
    Returns:
        dict: Dictionary containing "best_parameters" (dict of parameter values)
              and "profit" (best achieved profit).
    """
    optimizable_params = strategy_plugin.get_optimizable_params()
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
    # Register an individual generator that creates a list of attributes.
    def random_attr(param):
         name, low, high = param
         return random.uniform(low, high)
    
    toolbox.register("individual", lambda: creator.Individual([random_attr(param) for param in optimizable_params]))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Evaluation function: delegate to the plugin's evaluate_candidate method.
    def evaluate(individual):
         return strategy_plugin.evaluate_candidate(individual, base_data, hourly_predictions, daily_predictions, config)
    
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1.0, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Optimizer parameters
    random.seed(42)
    population_size = config.get("population_size", 20)
    num_generations = config.get("num_generations", 100)
    cxpb = config.get("crossover_probability", 0.5)
    mutpb = config.get("mutation_probability", 0.2)
    
    population = toolbox.population(n=population_size)
    print("Starting Genetic Algorithm Optimization...")
    print(f"Population Size: {population_size}, Generations: {num_generations}")
    
    # Use multiprocessing to speed up evaluation
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)
    
    # Evaluate the initial population.
    fitnesses = list(toolbox.map(toolbox.evaluate, population))
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
         invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
         fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
         for ind, fit in zip(invalid_ind, fitnesses):
              ind.fitness.values = fit
         population[:] = offspring
         fits = [ind.fitness.values[0] for ind in population]
         print(f"Generation {gen}: Max Profit = {max(fits):.2f}, Avg Profit = {sum(fits)/len(fits):.2f}")
    
    best_ind = tools.selBest(population, 1)[0]
    print("\nBest parameter set found:")
    best_params = {}
    for i, (name, low, high) in enumerate(optimizable_params):
         best_params[name] = best_ind[i]
         print(f"  {name} = {best_ind[i]:.4f}")
    print(f"Achieved Profit: {best_ind.fitness.values[0]:.2f}")
    
    pool.close()
    
    return {"best_parameters": best_params, "profit": best_ind.fitness.values[0]}
