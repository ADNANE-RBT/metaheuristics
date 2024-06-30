import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from fitness_functions import get_fitness_function
from report_generator import generate_report

def moth_flame_optimization(fitness_function, n, d, lb, ub, max_iterations, visualize=False):
    """
    Implements the Moth Flame Optimization Algorithm with visualization.
    
    Parameters:
    - fitness_function: The objective function to be optimized
    - n: Number of moths
    - d: Number of dimensions
    - lb: Lower bound of search space
    - ub: Upper bound of search space
    - max_iterations: Maximum number of iterations
    - visualize: Boolean to enable visualization
    
    Returns:
    - best_flame: The best solution found
    - best_fitness: The fitness value of the best solution
    - history: List of best fitness values over iterations
    - initial_positions: Initial positions of all moths
    - best_positions: Best positions in each iteration
    - best_scores: Best scores in each iteration
    """
    max_flames = n
    M = np.random.uniform(lb, ub, (n, d))
    initial_positions = M.copy()
    OM = np.array([fitness_function(m) for m in M])
    
    best_flame = M[np.argmin(OM)]
    best_fitness = np.min(OM)
    
    history = [best_fitness]
    best_positions = [best_flame]
    best_scores = [best_fitness]
    
    if visualize and d == 2:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        scatter = ax1.scatter(M[:, 0], M[:, 1], c='b', label='Moths')
        flame_scatter = ax1.scatter([], [], c='r', label='Flames')
        best_scatter = ax1.scatter([], [], c='g', s=100, label='Best')
        ax1.set_xlim(lb, ub)
        ax1.set_ylim(lb, ub)
        ax1.set_title('Moth and Flame Positions')
        ax1.legend()
        
        fitness_line, = ax2.plot([], [], 'b-')
        ax2.set_xlim(0, max_iterations)
        ax2.set_ylim(0, np.max(OM))
        ax2.set_title('Best Fitness over Iterations')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Fitness')
    
    for iteration in range(1, max_iterations + 1):
        if iteration == 1:
            F = np.sort(M, axis=0)
            OF = np.sort(OM)
        else:
            F = np.sort(np.vstack((F, M)), axis=0)
            OF = np.sort(np.concatenate((OF, OM)))
        
        
        num_flames = max(1, round(max_flames - iteration * ((max_flames - 1) / max_iterations)))
        F = F[:num_flames]
        OF = OF[:num_flames]
        
        a = -1 + iteration * ((-1) / max_iterations)
        
        for i in range(n):
            flame_index = min(i, num_flames - 1)
            for j in range(d):
                b = 1
                t = (a - 1) * np.random.random() + 1
                
                Di = np.abs(F[flame_index, j] - M[i, j])
                M[i, j] = Di * np.exp(b * t) * np.cos(2 * np.pi * t) + F[flame_index, j]
        
        M = np.clip(M, lb, ub)
        
        OM = np.array([fitness_function(m) for m in M])
        
        current_best = np.min(OM)
        if current_best < best_fitness:
            best_fitness = current_best
            best_flame = M[np.argmin(OM)]
        
        history.append(best_fitness)
        best_positions.append(best_flame)
        best_scores.append(best_fitness)
        
        if visualize and d == 2:
            scatter.set_offsets(M)
            flame_scatter.set_offsets(F[:num_flames])
            best_scatter.set_offsets([best_flame])
            
            fitness_line.set_data(range(iteration + 1), history)
            ax2.set_ylim(0, max(history))
            
            plt.pause(2)
        

    

    if visualize:
        plt.show()
    
    return best_flame, best_fitness, history, initial_positions, best_positions, best_scores

def shifted_rotated_sphere(x):
    shift = 100 * np.random.rand(len(x)) - 50  # Random shift in [-50, 50]
    A = np.random.randn(len(x), len(x))  # Random rotation matrix
    Q, _ = np.linalg.qr(A)  # Ensure the matrix is orthogonal
    y = np.dot(Q, x - shift)
    return np.sum(y**2) + 1  # Adding 1 to avoid zero fitness at optimum

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

    best_solution, best_fitness, history, initial_positions, best_positions, best_scores = moth_flame_optimization(
        fitness_function, n, d, lb, ub, max_iterations, visualize=False
    )
        # Generate the report
    generate_report(function_name, lb, ub, d, max_iterations, initial_positions, best_positions, best_scores, history, algo="MFO")

    all_results.append((function_name, history))



# Run visualizations if desired
if input("Do you want to see the optimization visualizations? (y/n): ").lower() == 'y':
    for function_name in function_names:
        fitness_function = get_fitness_function(function_name)
        moth_flame_optimization(fitness_function, n, d, lb, ub, max_iterations, visualize=True)

        
# for function_name in function_names:
#     fitness_function = get_fitness_function(function_name)

#     best_solution, best_fitness, history, initial_positions, best_positions, best_scores = moth_flame_optimization(
#         fitness_function, n, d, lb, ub, max_iterations, visualize=True
#     )

#     # Generate the report
#     generate_report(function_name, lb, ub, d, max_iterations, initial_positions, best_positions, best_scores, history)

#     # Plot the convergence history
#     plt.figure(figsize=(10, 5))
#     plt.plot(range(1, max_iterations + 2), [-h for h in history])
#     plt.title(f'Convergence History - {function_name.capitalize()} Function')
#     plt.xlabel('Iteration')
#     plt.ylabel('Best fitness')
#     plt.grid(True)
#     plt.show()

# Explanation of variables and functions:

# moth_flame_optimization: The main function implementing the algorithm.

# fitness_function: The objective function to be optimized.
# n: Number of moths in the population.
# d: Number of dimensions in the search space.
# lb: Lower bound of the search space.
# ub: Upper bound of the search space.
# max_iterations: Maximum number of iterations for the algorithm.


# M: Matrix representing moth positions (n x d).
# OM: Array of fitness values for each moth.
# F: Matrix of sorted flame positions.
# OF: Array of sorted flame fitness values.
# best_flame: The best solution found so far.
# best_fitness: The fitness value of the best solution.
# a: Parameter that decreases linearly from -1 to -2 over the course of iterations.
# b: Constant defining the shape of the logarithmic spiral.
# t: Random number in [-1, 1] used in the position update equation.
# Di: The distance between the i-th moth and the j-th flame.

# The algorithm works as follows:

# Initialize moth positions randomly within the search space.
# Evaluate the fitness of each moth.
# Sort flames based on their fitness.
# Update moth positions using the logarithmic spiral equation.
# Ensure moths stay within the search space bounds.
# Repeat steps 2-5 for the specified number of iterations.
# Return the best solution found.