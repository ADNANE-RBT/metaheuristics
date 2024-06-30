import os
import datetime
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


def generate_report(function_name, lb, ub, d, max_iterations, initial_positions, best_positions, best_scores, history, algo):
     # Create a folder for the algorithm
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"{algo}_DIR"
    os.makedirs(dir_name, exist_ok=True)

    # Create a folder for the report
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{algo}_Report_{function_name}_{timestamp}"
    report_folder_path = os.path.join(dir_name, folder_name)
    os.makedirs(report_folder_path, exist_ok=True)

    # Generate and save the convergence plot
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, max_iterations + 2), [-h for h in history])
    plt.title(f'Convergence History - {function_name.capitalize()} Function')
    plt.xlabel('Iteration')
    plt.ylabel('Best fitness')
    plt.grid(True)
    plt.savefig(os.path.join(report_folder_path, "convergence_plot.png"))
    plt.close()

    # Generate and save the best position plot
    # if d == 2:
    if len(best_positions[0]) == 2:  # Only for 2D
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot initial positions
        ax.scatter(initial_positions[:, 0], initial_positions[:, 1], c='blue', label='Initial Positions', alpha=0.5)
        
        # Create a color map for the progression of best positions
        n_positions = len(best_positions)
        colors = plt.cm.viridis(np.linspace(0, 1, n_positions))
        
        # Plot best positions with color gradient and arrows
        for i in range(n_positions - 1):
            ax.scatter(best_positions[i][0], best_positions[i][1], c=[colors[i]], s=100, 
                       label=f'Best Position {i+1}' if i == 0 else "")
            ax.annotate('', xy=(best_positions[i+1][0], best_positions[i+1][1]), 
                        xytext=(best_positions[i][0], best_positions[i][1]),
                        arrowprops=dict(arrowstyle="->", color=colors[i], lw=1.5))
        
        # Plot the final best position
        ax.scatter(best_positions[-1][0], best_positions[-1][1], c=[colors[-1]], s=150, 
                   label='Final Best Position', edgecolors='red', linewidths=2)
        
        ax.set_title(f'Moth Positions - {function_name.capitalize()} Function')
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.legend()
        ax.grid(True)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=0, vmax=n_positions-1))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, label="Iteration Progress", aspect=30, pad=0.08)
        
        plt.savefig(os.path.join(report_folder_path, "best_positions_plot.png"), dpi=300, bbox_inches='tight')
        plt.close()

    # Generate the report text file
    with open(os.path.join(report_folder_path, "MFO_Report.txt"), "w") as f:
        f.write("Moth-Flame Optimization Algorithm Report\n")
        f.write("=======================================\n\n")
        f.write(f"Fitness Function: {function_name}\n")
        f.write(f"Moth's Position Domain: [{lb}, {ub}]\n")
        f.write(f"Moth's Position Dimension: {d}\n")
        f.write(f"Number of Iterations: {max_iterations}\n\n")
        
        f.write("Initial Position of Moths:\n")
        for i, pos in enumerate(initial_positions):
            f.write(f"Moth {i+1}: {pos}\n")
        f.write("\n")
        
        f.write("Best moth's score and position in each iteration:\n")
        for i, (score, pos) in enumerate(zip(best_scores, best_positions)):
            f.write(f"Iteration {i+1}:\n")
            f.write(f"  Score: {score}\n")
            f.write(f"  Position: {pos}\n")
        
    print(f"Report generated in folder: {folder_name}")