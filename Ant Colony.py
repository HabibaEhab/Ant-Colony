import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import random


def load_data(filename):
    data = np.loadtxt(filename)
    return data[:, 1:], data[:, 0].astype(int)


def compute_distance_matrix(cities):
    return cdist(cities, cities, metric='euclidean')


def nearest_neighbor_heuristic(dist_matrix, start_city=0):
    n = len(dist_matrix)
    visited = [start_city]
    total_distance = 0

    for _ in range(n - 1):
        last = visited[-1]
        next_city = np.argmin([dist_matrix[last, j] if j not in visited else np.inf for j in range(n)])
        visited.append(next_city)
        total_distance += dist_matrix[last, next_city]

    total_distance += dist_matrix[visited[-1], visited[0]]
    return visited, total_distance


def initialize_pheromones(n, Lnn):
    tau_0 = 1 / (n * Lnn)
    return np.full((n, n), tau_0)


def remove_cycles(tour):
    """Removes cycles in the tour by keeping the first occurrence of a city."""
    seen = set()
    new_tour = []
    for city in tour:
        if city not in seen:
            new_tour.append(city)
            seen.add(city)
    new_tour.append(new_tour[0])  # Return to the starting city
    return new_tour


def acs_tsp(cities, n_ants=10, alpha=1, beta=2, rho=0.1, q0=0.9, iterations=20):
    n = len(cities)
    dist_matrix = compute_distance_matrix(cities)
    eta = 1 / (dist_matrix + np.eye(n) * 1e-10)  # Avoid division by zero

    Lnn = min(nearest_neighbor_heuristic(dist_matrix, i)[1] for i in range(n))
    pheromones = initialize_pheromones(n, Lnn)

    best_tour = None
    best_length = float('inf')

    for iteration in range(iterations):
        tours = []
        lengths = []

        start_cities = random.sample(range(n), min(n_ants, n))

        for start in start_cities:
            tour = [start]
            visited = set(tour)

            while len(tour) < n:
                current = tour[-1]
                unvisited = [j for j in range(n) if j not in visited]

                if random.random() < q0:
                    next_city = unvisited[
                        np.argmax([pheromones[current, j] * (eta[current, j] ** beta) for j in unvisited])]
                else:
                    probabilities = np.array([pheromones[current, j] * (eta[current, j] ** beta) for j in unvisited])
                    probabilities /= probabilities.sum()
                    next_city = np.random.choice(unvisited, p=probabilities)

                pheromones[current, next_city] = (1 - rho) * pheromones[current, next_city] + rho * (1 / (n * Lnn))
                tour.append(next_city)
                visited.add(next_city)

            tour.append(tour[0])
            tour = remove_cycles(tour)  # Ensure no cycles
            length = sum(dist_matrix[tour[i], tour[i + 1]] for i in range(len(tour) - 1))
            tours.append(tour)
            lengths.append(length)

            if length < best_length:
                best_tour = tour
                best_length = length

        delta_tau = 1 / best_length
        for i in range(len(best_tour) - 1):
            pheromones[best_tour[i], best_tour[i + 1]] = (1 - rho) * pheromones[
                best_tour[i], best_tour[i + 1]] + rho * delta_tau

    return best_tour, best_length, cities


def plot_cities(cities):
    plt.figure(figsize=(8, 6))
    plt.scatter(cities[:, 0], cities[:, 1], c='red', marker='o')
    for i, (x, y) in enumerate(cities):
        plt.text(x, y, str(i), fontsize=8, ha='right')
    plt.title("Cities Visualization")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.grid(True)
    plt.show()


def plot_results(cities, tours, lengths, params):
    plt.figure(figsize=(12, 6))
    plt.scatter(cities[:, 0], cities[:, 1], c='red', marker='o')
    for i, (x, y) in enumerate(cities):
        plt.text(x, y, str(i), fontsize=8, ha='right')

    for i, tour in enumerate(tours):
        tour_x = cities[tour, 0]
        tour_y = cities[tour, 1]
        plt.plot(tour_x, tour_y, label=f'Length: {lengths[i]:.2f}', alpha=0.7)

    plt.title(
        f'TSP Solution with ACS (α={params["alpha"]}, β={params["beta"]}, ρ={params["rho"]}, Ants={params["n_ants"]})')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def get_user_parameters():
    print("Enter ACS parameters (press Enter for default values):")
    alpha = float(input(f"Alpha (default 1): ") or 1)
    beta = float(input(f"Beta (default 2): ") or 2)
    rho = float(input(f"Rho (default 0.1): ") or 0.1)
    n_ants = int(input(f"Number of ants (default 10): ") or 10)
    iterations = int(input(f"Number of iterations (default 20): ") or 20)
    return alpha, beta, rho, n_ants, iterations


if __name__ == "__main__":
    cities, city_ids = load_data("TSPDATA.txt")

    # Plot just the cities first
    plot_cities(cities)

    # Get parameters from user
    alpha, beta, rho, n_ants, iterations = get_user_parameters()

    # Run ACS with user parameters
    best_tour, best_length, _ = acs_tsp(
        cities,
        n_ants=n_ants,
        alpha=alpha,
        beta=beta,
        rho=rho,
        iterations=iterations
    )

    print(f"Best tour length: {best_length:.4f}")
    plot_results(
        cities,
        [best_tour],
        [best_length],
        {'alpha': alpha, 'beta': beta, 'rho': rho, 'n_ants': n_ants}
    )