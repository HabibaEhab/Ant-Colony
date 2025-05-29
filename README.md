# Ant Colony System (ACS) for Traveling Salesman Problem (TSP)

This project implements an Ant Colony System (ACS) metaheuristic algorithm to solve the Traveling Salesman Problem (TSP), a classic combinatorial optimization challenge. The algorithm simulates the behavior of ants depositing pheromones to probabilistically build near-optimal routes visiting all cities once.

## Features
- Loads city coordinates from a data file.
- Computes Euclidean distance matrix between cities.
- Implements the ACS with pheromone updating, heuristic information, and probabilistic city selection.
- Includes parameter tuning for alpha (pheromone importance), beta (heuristic importance), rho (pheromone evaporation), number of ants, and iterations.
- Visualizes city locations and the best-found tour using Matplotlib.
- Interactive user interface for inputting ACS parameters.

## How It Works
- Reads city data and calculates distance matrix.
- Uses nearest neighbor heuristic to estimate initial route length.
- Initializes pheromone levels based on the heuristic.
- For each iteration, each ant constructs a tour by probabilistically selecting the next city based on pheromone levels and heuristic desirability.
- Updates pheromones locally during tour construction and globally after iteration completion.
- Keeps track of and outputs the best tour found.

## Usage
1. Prepare a data file `TSPDATA.txt` containing city IDs and coordinates.
2. Run the Python script.
3. View the plot of city locations.
4. Input algorithm parameters when prompted or use defaults.
5. The script outputs the best route length and displays a visualization of the best tour.

## Requirements
- Python 3.x
- numpy
- matplotlib
- scipy


