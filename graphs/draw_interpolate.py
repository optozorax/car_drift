import json
import matplotlib.pyplot as plt
from typing import List, Tuple

def load_and_process_data(filename: str) -> List[Tuple[List[float], List[float]]]:
    with open(filename, 'r') as f:
        data = json.load(f)
    
    # Split x and y coordinates
    plots = []
    for plot in data:
        x = [point[0] for point in plot]
        y = [point[1] for point in plot]
        plots.append((x, y))
    return plots

def calculate_statistics(plots: List[Tuple[List[float], List[float]]], num_points: int = 1000):
    # Find global x range
    min_x = min(min(plot[0]) for plot in plots)
    max_x = max(max(plot[0]) for plot in plots)
    
    # Create reference x points
    reference_x = [min_x + (max_x - min_x) * i / (num_points - 1) for i in range(num_points)]
    
    # Interpolate y values
    interpolated_y = []
    for x_coords, y_coords in plots:
        # Linear interpolation
        new_y = []
        for x in reference_x:
            # Find surrounding points
            for i in range(len(x_coords) - 1):
                if x_coords[i] <= x <= x_coords[i + 1]:
                    # Linear interpolation formula
                    ratio = (x - x_coords[i]) / (x_coords[i + 1] - x_coords[i])
                    y = y_coords[i] + ratio * (y_coords[i + 1] - y_coords[i])
                    new_y.append(y)
                    break
            else:
                # If x is outside range, use nearest value
                if x < x_coords[0]:
                    new_y.append(y_coords[0])
                else:
                    new_y.append(y_coords[-1])
        interpolated_y.append(new_y)
    
    # Calculate statistics
    n = len(interpolated_y)
    mean_y = [sum(y[i] for y in interpolated_y) / n for i in range(num_points)]
    
    # Calculate percentiles
    sorted_y = [[y[i] for y in interpolated_y] for i in range(num_points)]
    for i in range(len(sorted_y)):
        sorted_y[i].sort()
    
    p25_y = [sorted_y[i][int(n * 0.25)] for i in range(num_points)]
    p75_y = [sorted_y[i][int(n * 0.75)] for i in range(num_points)]
    
    return reference_x, mean_y, p25_y, p75_y

def plot_data(filename: str):
    # Load data
    plots = load_and_process_data(filename)
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot individual trajectories
    for x, y in plots:
        plt.plot(x, y, 'b-', alpha=0.1, linewidth=1)
    
    # Calculate and plot statistics
    x, mean_y, p25_y, p75_y = calculate_statistics(plots)
    
    # Plot mean and percentiles
    plt.plot(x, mean_y, 'r-', linewidth=2, label='Mean')
    plt.fill_between(x, p25_y, p75_y, color='r', alpha=0.2, label='25-75 percentile')
    
    # Customize plot
    plt.grid(True)
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Plots with Statistics')
    
    plt.show()

# Usage
plot_data('interpolation_data.json')
