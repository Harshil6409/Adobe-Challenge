import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

# Helper functions
def reflect(points, line):
    """ Reflect points over a given line. """
    a, b = line
    reflected_points = []
    for point in points:
        # Calculate the reflection of point over the line
        d = np.dot(point - a, b - a) / np.dot(b - a, b - a)
        reflected_point = 2 * (a + d * (b - a)) - point
        reflected_points.append(reflected_point)
    return np.array(reflected_points)

def is_symmetric(original, reflected):
    """ Check if the original points and reflected points are the same. """
    return np.allclose(np.sort(original, axis=0), np.sort(reflected, axis=0))

def fit_bezier(points, degree=3):
    """ Fit a Bezier curve to a set of points. """
    n = len(points) - 1
    
    def bezier(t, control_points):
        return sum(
            (np.math.comb(n, i) * (1-t)**(n-i) * t**i) * control_points[i]
            for i in range(len(control_points))
        )

    def loss(control_points):
        control_points = control_points.reshape(-1, 2)
        t_values = np.linspace(0, 1, len(points))
        bezier_points = np.array([bezier(t, control_points) for t in t_values])
        return np.sum(np.linalg.norm(bezier_points - points, axis=1))

    initial_control_points = np.linspace(points[0], points[-1], degree + 1)
    result = minimize(loss, initial_control_points.flatten())
    return result.x.reshape(-1, 2)

# Main function
def find_symmetry(points):
    num_points = len(points)
    lines = []

    # Iterate through all pairs of points
    for i in range(num_points):
        for j in range(i+1, num_points):
            line = (points[i], points[j])
            reflected_points = reflect(points, line)
            if is_symmetric(points, reflected_points):
                lines.append(line)
                print(f"Symmetry found with line: {line}")

    return lines

def plot_shape_with_symmetry(points, symmetry_lines):
    plt.figure()
    plt.plot(points[:, 0], points[:, 1], 'bo-', label='Shape')
    for line in symmetry_lines:
        plt.plot([line[0][0], line[1][0]], [line[0][1], line[1][1]], 'r--', label='Symmetry Line')
    plt.axis('equal')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Example shape
    points = np.array([
        [1, 1],
        [2, 3],
        [4, 3],
        [5, 1],
        [4, -1],
        [2, -1]
    ])
    
    # Find symmetry lines
    symmetry_lines = find_symmetry(points)
    
    # Fit Bezier curve (example)
    fitted_curve = fit_bezier(points)
    print("Fitted Bezier curve control points:", fitted_curve)
    
    # Plot results
    plot_shape_with_symmetry(points, symmetry_lines)
