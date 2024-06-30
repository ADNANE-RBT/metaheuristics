import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from fitness_functions import get_fitness_function
from report_generator import generate_report

def honey_badger_optimization(fitness_function, n, d, lb, ub, max_iterations, visualize=False):
    """
    Implements the Honey Badger Optimization Algorithm with visualization.
    
    Parameters:
    - fitness_function: The objective function to be optimized
    - n: Number of honey badgers
    - d: Number of dimensions
    - lb: Lower bound of search space
    - ub: Upper bound of search space
    - max_iterations: Maximum number of iterations
    - visualize: Boolean to enable visualization
    
    Returns:
    - best_solution: The best solution found
    - best_fitness: The fitness value of the best solution
    - history: List of best fitness values over iterations
    - initial_positions: Initial positions of all honey badgers
    - best_positions: Best positions in each iteration
    - best_scores: Best scores in each iteration
    """
    population = np.random.uniform(lb, ub, (n, d))
    initial_positions = population.copy()
    fitness = np.array([fitness_function(ind) for ind in population])
    
    best_solution = population[np.argmin(fitness)]
    best_fitness = np.min(fitness)
    
    history = [best_fitness]
    best_positions = [best_solution]
    best_scores = [best_fitness]
    
    if visualize and d == 2:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        scatter = ax1.scatter(population[:, 0], population[:, 1], c='b', label='Honey Badgers')
        best_scatter = ax1.scatter([], [], c='r', s=100, label='Best')
        ax1.set_xlim(lb, ub)
        ax1.set_ylim(lb, ub)
        ax1.set_title('Honey Badger Positions')
        ax1.legend()
        
        fitness_line, = ax2.plot([], [], 'b-')
        ax2.set_xlim(0, max_iterations)
        ax2.set_ylim(0, np.max(fitness))
        ax2.set_title('Best Fitness over Iterations')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Fitness')

    for iteration in range(max_iterations):
        for i in range(n):
            # Foraging behavior
            if np.random.rand() < 0.5:
                r1, r2 = np.random.rand(2)
                new_position = population[i] + r1 * (best_solution - r2 * population[i])
            # Aggressive behavior
            else:
                r3, r4 = np.random.rand(2)
                random_badger = population[np.random.randint(n)]
                new_position = population[i] + r3 * (random_badger - r4 * population[i])

            # Boundary check
            new_position = np.clip(new_position, lb, ub)

            # Update if better
            new_fitness = fitness_function(new_position)
            if new_fitness < fitness[i]:
                population[i] = new_position
                fitness[i] = new_fitness

            # Update best solution
            if new_fitness < best_fitness:
                best_solution = new_position
                best_fitness = new_fitness

        history.append(best_fitness)
        best_positions.append(best_solution)
        best_scores.append(best_fitness)
        
        if visualize and d == 2:
            scatter.set_offsets(population)
            best_scatter.set_offsets([best_solution])
            
            fitness_line.set_data(range(iteration + 2), history)
            ax2.set_ylim(0, max(history))
            
            plt.pause(0.1)

    if visualize:
        plt.show()
    
    return best_solution, best_fitness, history, initial_positions, best_positions, best_scores

# Example usage
d = 2   # Number of dimensions (2D space)
n = 50  # Number of honey badgers
lb = -100  # Lower bound of the space
ub = 100  # Upper bound of the space
max_iterations = 100

# Define a list of fitness function names
function_names = [
    'f1_sphere', 'f2_ellipsoidal', 'f3_rastrigin', 'f4_buche_rastrigin', 'f5_linear_slope',
    'f6_attractive_sector', 'f7_step_ellipsoidal', 'f8_rosenbrock', 'f9_rosenbrock_rotated',
    'f10_ellipsoidal_rotated', 'f11_discus', 'f12_bent_cigar', 'f13_sharp_ridge', 'f14_different_powers',
    'f15_rastrigin_rotated', 'f16_weierstrass', 'f17_schaffers_f7', 'f18_schaffers_f7_ill_conditioned',
    'f19_composite_griewank_rosenbrock', 'f20_schwefel', 'f21_gallagher_gaussian_101me',
    'f22_gallagher_gaussian_21hi', 'f23_katsuura', 'f24_lunacek_bi_rastrigin'
]

# Run optimizations and generate reports
all_results = []
for function_name in function_names:
    fitness_function = get_fitness_function(function_name)

    best_solution, best_fitness, history, initial_positions, best_positions, best_scores = honey_badger_optimization(
        fitness_function, n, d, lb, ub, max_iterations, visualize=False
    )
    
    # Generate the report
    generate_report(function_name, lb, ub, d, max_iterations, initial_positions, best_positions, best_scores, history, algo="HBO")

    all_results.append((function_name, history))

# Run visualizations if desired
if input("Do you want to see the optimization visualizations? (y/n): ").lower() == 'y':
    for function_name in function_names:
        fitness_function = get_fitness_function(function_name)
        honey_badger_optimization(fitness_function, n, d, lb, ub, max_iterations, visualize=True)