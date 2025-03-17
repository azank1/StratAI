import pygad

def fitness_func(solution, solution_idx, ga_instance):
    # Example: just return 0 for demonstration
    return 0.0

ga_instance = pygad.GA(
    num_generations=5,
    num_parents_mating=5,  # <-- REQUIRED
    sol_per_pop=20,
    num_genes=2,
    fitness_func=fitness_func,
    gene_space=[
        {'low': 5, 'high': 30},
        {'low': 31, 'high': 100}
    ]
)

ga_instance.run()
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Best solution:", solution)
print("Fitness:", solution_fitness)
