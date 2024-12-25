import numpy as np
import random
import json
import pickle


class AlgorithmState:
    def __init__(self, iteration, population, fitness, best_solution, best_fitness):
        self.iteration = iteration
        self.population = population
        self.fitness = fitness
        self.best_solution = best_solution
        self.best_fitness = best_fitness


def save_state(filename, state):
    with open(filename, 'wb') as f:
        pickle.dump(state, f)


def load_state(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def benchmark_algorithm(algorithm_function, algorithm_name, functions, max_iterations, population_sizes, trials):
    results = []

    for function_name, (objective_function, properties) in functions.items():
        for dimensions in properties()['dimensions']:
            for pop_size in population_sizes:
                for max_iter in max_iterations:
                    best_fitness_list = []
                    for trial in range(trials):
                        lower_boundary = properties()['lower_boundary']
                        upper_boundary = properties()['upper_boundary']
                        best_solution, best_fitness = algorithm_function(
                            objective_function, dimensions, lower_boundary, upper_boundary, pop_size, max_iter
                        )
                        best_fitness_list.append(best_fitness)

                    avg_fitness = np.mean(best_fitness_list)
                    std_fitness = np.std(best_fitness_list)
                    results.append({
                        'algorithm': algorithm_name,
                        'function': function_name,
                        'dimensions': dimensions,
                        'population_size': pop_size,
                        'max_iterations': max_iter,
                        'average_fitness': avg_fitness,
                        'std_dev_fitness': std_fitness,
                        'trials': trials
                    })

    benchmark_file = f'benchmark_{algorithm_name.lower().replace(" ", "_")}.json'
    with open(benchmark_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Benchmark results saved to {benchmark_file}")


def jellyfish_search(objective_function, dimensions, lower_boundary, upper_boundary, population_size, max_iterations):
    jellyfish = np.random.uniform(lower_boundary, upper_boundary, (population_size, dimensions))
    fitness = np.array([objective_function(position) for position in jellyfish])
    best_solution = jellyfish[np.argmin(fitness)]
    best_fitness = np.min(fitness)

    beta = 3.0
    gamma = 0.1

    for iteration in range(max_iterations):
        for current in range(population_size):
            time_control = abs((1 - iteration / max_iterations) * (2 * np.random.rand() - 1))

            if time_control >= 0.5:
                micro = np.mean(jellyfish, axis=0)
                ocean_current = best_solution - beta * np.random.rand() * micro
                jellyfish[current] = jellyfish[current] + np.random.rand() * ocean_current
            else:
                if np.random.rand() > 1 - time_control:
                    jellyfish[current] = jellyfish[current] + gamma * np.random.rand() * (upper_boundary - lower_boundary)
                else:
                    random_number = np.random.randint(population_size)
                    direction = (
                        jellyfish[random_number] - jellyfish[current]
                        if fitness[current] >= fitness[random_number]
                        else jellyfish[current] - jellyfish[random_number]
                    )
                    step = np.random.rand() * direction
                    jellyfish[current] += step

            jellyfish[current] = np.clip(jellyfish[current], lower_boundary, upper_boundary)
            new_fitness = objective_function(jellyfish[current])

            if new_fitness <= fitness[current]:
                fitness[current] = new_fitness
                if new_fitness <= best_fitness:
                    best_solution = jellyfish[current].copy()
                    best_fitness = new_fitness

    return best_solution, best_fitness


def artificial_bee_colony(objective_function, lower_boundary, upper_boundary, dimensions, population_size, max_iterations, limit=10):
    dimensions = int(dimensions)
    population_size = int(population_size)
    max_iterations = int(max_iterations)

    population = np.random.uniform(lower_boundary, upper_boundary, (population_size, dimensions))
    fitness = np.array([
        1 / (1 + objective_function(solution)) if objective_function(solution) >= 0
        else 1 + abs(objective_function(solution))
        for solution in population
    ])
    best_solution = population[np.argmax(fitness)]
    best_fitness = objective_function(best_solution)
    failures = np.zeros(population_size, dtype=int)

    def generate_new_solution(index):
        dimension = random.randint(0, dimensions - 1)
        neighbor = random.randint(0, population_size - 1)
        while neighbor == index:
            neighbor = random.randint(0, population_size - 1)

        new_solution = population[index].copy()
        diff = new_solution[dimension] - population[neighbor][dimension]
        new_solution[dimension] += (random.random() - 0.5) * 2 * diff
        new_solution = np.clip(new_solution, lower_boundary, upper_boundary)

        new_fitness = (
            1 / (1 + objective_function(new_solution))
            if objective_function(new_solution) >= 0
            else 1 + abs(objective_function(new_solution))
        )
        if new_fitness > fitness[index]:
            population[index] = new_solution
            fitness[index] = new_fitness
            failures[index] = 0
        else:
            failures[index] += 1

    for iteration in range(max_iterations):
        for i in range(population_size):
            generate_new_solution(i)

        probabilities = fitness / fitness.sum()
        for i in range(population_size):
            if random.random() < probabilities[i]:
                generate_new_solution(i)

        for i in range(population_size):
            if failures[i] > limit:
                population[i] = np.random.uniform(lower_boundary, upper_boundary, dimensions)
                fitness[i] = (
                    1 / (1 + objective_function(population[i]))
                    if objective_function(population[i]) >= 0
                    else 1 + abs(objective_function(population[i]))
                )
                failures[i] = 0

        current_best = population[np.argmax(fitness)]
        current_fitness = objective_function(current_best)
        if current_fitness < best_fitness:
            best_solution = current_best
            best_fitness = current_fitness

    return best_solution, best_fitness


