# import os
# import numpy as np
# from abc import ABC, abstractmethod
# from typing import List, Type, TextIO, Iterable
# from enum import Enum
# import random
# import sys
#
#
# ###############     TEST FUNCTIONS    ###############
#
# # Interface for test functions:
# class FunctionInterface(ABC):
#     max_iterations = [50, 100, 500, 1000]
#     population_sizes = [30, 50, 100]
#     trials = 10
#
#     # Abstract members:
#     @property
#     @abstractmethod
#     def name(self):
#         pass
#
#     @property
#     @abstractmethod
#     def minimum(self):
#         pass
#
#     @property
#     @abstractmethod
#     def dimensions(self) -> Iterable[int]:
#         pass
#
#     @property
#     @abstractmethod
#     def lower_boundary(self):
#         pass
#
#     @property
#     @abstractmethod
#     def upper_boundary(self):
#         pass
#
#     @staticmethod
#     @abstractmethod
#     def function(x):
#         pass
#
#
# # Functions itself:
# class Rastrigin(FunctionInterface):
#     name = "Rastrigin"
#     minimum = 0.0
#     dimensions = [2, 5, 10, 30]
#     lower_boundary = -5.12
#     upper_boundary = 5.12
#
#     @staticmethod
#     def function(x):
#         A = 10
#         return A * len(x) + np.sum(x ** 2 - A * np.cos(2 * np.pi * x))
#
#
# class Rosenbrock(FunctionInterface):
#     name = "Rosenbrock"
#     minimum = 0.0
#     dimensions = [2, 5, 10, 30]
#     lower_boundary = -5.0
#     upper_boundary = 5.0
#
#     @staticmethod
#     def function(x):
#         return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)
#
#
# class Sphere(FunctionInterface):
#     name = "Sphere"
#     minimum = 0.0
#     dimensions = [2, 5, 10, 30, 50]
#     lower_boundary = -5.12
#     upper_boundary = 5.12
#
#     @staticmethod
#     def function(x):
#         return np.sum(x ** 2)
#
#
# class Beale(FunctionInterface):
#     name = "Beale"
#     minimum = 0.0
#     dimensions = [2]
#     lower_boundary = -4.5
#     upper_boundary = 4.5
#
#     @staticmethod
#     def function(x):
#         x1, x2 = x[0], x[1]
#         term1 = (1.5 - x1 + x1 * x2) ** 2
#         term2 = (2.25 - x1 + x1 * x2 ** 2) ** 2
#         term3 = (2.625 - x1 + x1 * x2 ** 3) ** 2
#         return term1 + term2 + term3
#
#
# class Bukin(FunctionInterface):
#     name = "Bukin"
#     minimum = 0.0
#     dimensions = [2]
#     lower_boundary = -15.0
#     upper_boundary = 3.0
#
#     @staticmethod
#     def function(x):
#         x1, x2 = x[0], x[1]
#         x1 = max(min(x1, -5.0), -15.0)
#         x2 = max(min(x2, 3.0), -3.0)
#         term1 = 100 * np.sqrt(np.abs(x2 - 0.01 * x1 ** 2))
#         term2 = 0.01 * np.abs(x1 + 10)
#         return term1 + term2
#
#
# class Himmelblaus(FunctionInterface):
#     name = "Himmelblaus"
#     minimum = 0.0
#     dimensions = [2]
#     lower_boundary = -5.0
#     upper_boundary = 5.0
#
#     @staticmethod
#     def function(x):
#         x1, x2 = x[0], x[1]
#         term1 = (x1 ** 2 + x2 - 11) ** 2
#         term2 = (x1 + x2 ** 2 - 7) ** 2
#         return term1 + term2
#
#
# ###############     BENCHMARK CLASSES     ###############
# class ResultType(Enum):
#     BEST = 0,
#     WORST = 1
#
#
# class Benchmark:
#     # Properties:
#     current_benchmark_function: FunctionInterface = None
#     current_dimension = None
#     current_max_iterations = None
#     current_population_size = None
#     current_best_solution = None
#     current_best_fitness: float = np.inf
#     current_worst_solution = None
#     current_worst_fitness: float = -np.inf
#
#     # Benchmark methods:
#     @staticmethod
#     def run(benchmark_functions: List[Type[FunctionInterface]]):
#         file_path = '../../../PycharmProjects/jfs/benchmarkJSO.csv'
#         with open(file_path, 'a', encoding='utf-8') as file:
#             if os.stat(file_path).st_size != 0:
#                 file.write('\n\n')
#             file.write('Algorithm;'
#                        'FunctionName;'
#                        'Trials;'
#                        'Dimension;'
#                        'MaxIterations;'
#                        'PopulationSize;'
#                        'LowerBoundary;'
#                        'UpperBoundary;'
#                        'SearchedMinimum;'
#                        'BestFitness;'
#                        'BestSolution;'
#                        'ResultType\n')
#             Benchmark.benchmark(benchmark_functions, file)
#
#     @staticmethod
#     def benchmark(benchmark_functions: List[Type[FunctionInterface]], file: TextIO):
#         for benchmark_function in benchmark_functions:
#             Benchmark.current_benchmark_function = benchmark_function
#             for dimension in benchmark_function.dimensions:
#                 Benchmark.current_dimension = dimension
#                 for max_iterations in benchmark_function.max_iterations:
#                     Benchmark.current_max_iterations = max_iterations
#                     for population_size in benchmark_function.population_sizes:
#                         Benchmark.current_population_size = population_size
#                         for _ in range(benchmark_function.trials):
#                             trial_solution, trial_fitness = jellyfish_search(benchmark_function.function,
#                                                                              dimension,
#                                                                              benchmark_function.lower_boundary,
#                                                                              benchmark_function.upper_boundary,
#                                                                              population_size,
#                                                                              max_iterations)
#
#                             if Benchmark.current_best_fitness > trial_fitness:
#                                 Benchmark.current_best_solution = trial_solution
#                                 Benchmark.current_best_fitness = trial_fitness
#
#                             if Benchmark.current_worst_fitness < trial_fitness:
#                                 Benchmark.current_worst_solution = trial_solution
#                                 Benchmark.current_worst_fitness = trial_fitness
#
#                         Benchmark.write_to_file(file, ResultType.BEST)
#                         Benchmark.write_to_file(file, ResultType.WORST)
#                         Benchmark.print_on_screen(ResultType.BEST)
#                         Benchmark.print_on_screen(ResultType.WORST)
#
#                         Benchmark.current_best_solution = None
#                         Benchmark.current_best_fitness = np.inf
#                         Benchmark.current_worst_solution = None
#                         Benchmark.current_worst_fitness = -np.inf
#
#     @staticmethod
#     def write_to_file(file: TextIO, result_type: ResultType):
#         file.write(f'JSO;'
#                    f'{Benchmark.current_benchmark_function.name};'
#                    f'{Benchmark.current_benchmark_function.trials};'
#                    f'{Benchmark.current_dimension};'
#                    f'{Benchmark.current_max_iterations};'
#                    f'{Benchmark.current_population_size};'
#                    f'{Benchmark.current_benchmark_function.lower_boundary};'
#                    f'{Benchmark.current_benchmark_function.upper_boundary};'
#                    f'{Benchmark.current_benchmark_function.minimum};')
#
#         if result_type == ResultType.BEST:
#             Benchmark.write_best_to_file(file)
#         else:
#             Benchmark.write_worst_to_file(file)
#
#     @staticmethod
#     def write_best_to_file(file: TextIO):
#         current_best_solution_str = f"({', '.join(map(str, Benchmark.current_best_solution))})"
#         file.write(f'{Benchmark.current_best_fitness};'
#                    f'{current_best_solution_str};'
#                    f'BEST\n')
#
#     @staticmethod
#     def write_worst_to_file(file: TextIO):
#         current_worst_solution_str = f"({', '.join(map(str, Benchmark.current_worst_solution))})"
#         file.write(f'{Benchmark.current_worst_fitness};'
#                    f'{current_worst_solution_str};'
#                    f'WORST\n')
#
#     # Print on screen methods:
#     @staticmethod
#     def print_on_screen(result_type: ResultType):
#         print(f'Algorithm: JSO\n'
#               f'Function: {Benchmark.current_benchmark_function.name}\n'
#               f'Trials = {Benchmark.current_benchmark_function.trials}\n'
#               f'Dimensions = {Benchmark.current_dimension}\n'
#               f'Max iterations = {Benchmark.current_max_iterations}\n'
#               f'Population size = {Benchmark.current_population_size}\n'
#               f'Domain: From {Benchmark.current_benchmark_function.lower_boundary} to {Benchmark.current_benchmark_function.upper_boundary}\n'
#               f'Searched minimum = {Benchmark.current_benchmark_function.minimum}')
#
#         if result_type == ResultType.BEST:
#             Benchmark.print_best_on_screen()
#         else:
#             Benchmark.print_worst_on_screen()
#
#     @staticmethod
#     def print_best_on_screen():
#         print(f'Best fitness = {Benchmark.current_best_fitness}\n'
#               f'Best solution = {Benchmark.current_best_solution}\n'
#               f'Result type = BEST\n')
#
#     @staticmethod
#     def print_worst_on_screen():
#         print(f'Worst fitness = {Benchmark.current_worst_fitness}\n'
#               f'Worst solution = {Benchmark.current_worst_solution}\n'
#               f'Result type = WORST\n')
#
#
# def jellyfish_search(objective_function, dimensions, lower_boundary, upper_boundary, population_size, max_iterations):
#     # Initialize the population:
#     jellyfish = np.random.uniform(lower_boundary, upper_boundary, (population_size, dimensions))
#     fitness = np.array([objective_function(position) for position in jellyfish])
#
#     # Calculate start values:
#     best_jellyfish = jellyfish[np.argmin(fitness)]
#     best_fitness = np.min(fitness)
#
#     beta = 3.0
#     gamma = 0.1
#
#     for iteration in range(max_iterations):
#         for current in range(population_size):
#             # Time control function:
#             time_control = abs((1 - iteration / max_iterations) * (2 * np.random.rand() - 1))
#
#             if time_control >= 0.5:
#                 # Jellyfish follows ocean current:
#                 micro = np.mean(jellyfish, axis=0)
#                 ocean_current = best_jellyfish - beta * np.random.rand() * micro
#                 jellyfish[current] = jellyfish[current] + np.random.rand() * ocean_current
#             else:
#                 if np.random.rand() > 1 - time_control:
#                     # Passive jellyfish motion:
#                     jellyfish[current] = jellyfish[current] + gamma * np.random.rand() * (
#                             upper_boundary - lower_boundary)
#                 else:
#                     # Active jellyfish motion:
#                     random_number = np.random.randint(population_size)
#                     direction = jellyfish[random_number] - jellyfish[current] if fitness[current] >= fitness[
#                         random_number] else jellyfish[current] - jellyfish[random_number]
#                     step = np.random.rand() * direction
#                     jellyfish[current] += step
#
#             # Boundary control:
#             jellyfish[current] = np.clip(jellyfish[current], lower_boundary, upper_boundary)
#
#             # Evaluate new position:
#             new_fitness = objective_function(jellyfish[current])
#
#             if new_fitness <= fitness[current]:
#                 fitness[current] = new_fitness
#                 if new_fitness <= best_fitness:
#                     best_jellyfish = jellyfish[current].copy()
#                     best_fitness = new_fitness
#
#         # print(f'Iteration {iteration + 1} -> best_fitness = {best_fitness}')
#
#     return best_jellyfish, best_fitness
#
#
# ###############     MAIN     ###############
# if __name__ == '__main__':
#     benchmark_functions: List[Type[FunctionInterface]] = [Rastrigin, Rosenbrock, Sphere, Beale, Bukin, Himmelblaus]
#
#     Benchmark.run(benchmark_functions)
#
# class Config:
#     def __init__(self, argv):
#         # Ustawianie domyślnych parametrów lub pobieranie ich z argv
#         self.DOLNA_GRANICA = -10
#         self.GORNA_GRANICA = 10
#         self.LICZBA_ZRODEL = 20
#         self.WYMIAR = 5
#         self.LIMIT = 100  # Limit dla pszczół zwiadowczych
#         self.MAKSYMALNA_LICZBA_ITERACJI = 1000
#
#
# class AlgorytmABC:
#     def __init__(self, conf):
#         self.conf = conf
#         self.rozwiazania = None
#         self.dopasowanie = None
#         self.niepowodzenia = np.zeros(self.conf.LICZBA_ZRODEL)
#         self.najlepszy = 0
#         self.naj_rozwiazanie = None
#         self.prawdopodobienstwa = None
#         self.iteracje = 0
#         self.cykl = 0
#
#     def kalkulator_dopasowania(self, funkcja):
#         return 1 / (1 + funkcja) if funkcja >= 0 else 1 + np.abs(funkcja)
#
#     def kalkulator_funkcji(self, rozwiazanie):
#         return np.sum(rozwiazanie ** 2)
#
#     def inicjalizuj(self):
#         self.rozwiazania = self.conf.DOLNA_GRANICA + np.random.rand(self.conf.LICZBA_ZRODEL, self.conf.WYMIAR) * (
#                     self.conf.GORNA_GRANICA - self.conf.DOLNA_GRANICA)
#         self.dopasowanie = np.array([self.kalkulator_dopasowania(self.kalkulator_funkcji(i)) for i in self.rozwiazania])
#         self.naj_rozwiazanie = np.copy(self.rozwiazania[np.argmax(self.dopasowanie)])
#         self.najlepszy = self.kalkulator_funkcji(self.naj_rozwiazanie)
#
#     def generuj_nowe_rozwiazanie(self, indeks):
#         losowy_wymiar = int(self.conf.WYMIAR * random.random())
#         losowy_sasiad = int(self.conf.LICZBA_ZRODEL * random.random())
#
#         while losowy_sasiad == indeks:
#             losowy_sasiad = int(self.conf.LICZBA_ZRODEL * random.random())
#
#         nowe_rozwiazanie = np.copy(self.rozwiazania[indeks])
#         stare_rozwiazanie = nowe_rozwiazanie[losowy_wymiar]
#         nowe_rozwiazanie[losowy_wymiar] = max(
#             min(stare_rozwiazanie + (stare_rozwiazanie - self.rozwiazania[losowy_sasiad, losowy_wymiar]) * (
#                         (random.random() - 0.5) * 2),
#                 self.conf.GORNA_GRANICA),
#             self.conf.DOLNA_GRANICA
#         )
#         nowe_dopasowanie = self.kalkulator_dopasowania(self.kalkulator_funkcji(nowe_rozwiazanie))
#
#         if nowe_dopasowanie > self.dopasowanie[indeks]:
#             self.rozwiazania[indeks] = nowe_rozwiazanie
#             self.dopasowanie[indeks] = nowe_dopasowanie
#             self.niepowodzenia[indeks] = 0
#         else:
#             self.niepowodzenia[indeks] += 1
#
#         self.zwieksz_liczbe_iteracji()
#
#     def wyslij_pszczoly_zatrudnione(self):
#         i = 0
#         while i < self.conf.LICZBA_ZRODEL and not self.warunek_stopu():
#             self.generuj_nowe_rozwiazanie(i)
#             i += 1
#
#     def kalkulator_prawdopodobienstwa(self):
#         maxDopasowanie = np.max(self.dopasowanie)
#         self.prawdopodobienstwa = 0.9 * self.dopasowanie / maxDopasowanie + 0.1
#
#     def wyslij_pszczoly_obserwujace(self):
#         i = 0
#         j = 0
#         while j < self.conf.LICZBA_ZRODEL and not self.warunek_stopu():
#             r = random.random()
#             if r < self.prawdopodobienstwa[i]:
#                 j += 1
#                 self.generuj_nowe_rozwiazanie(i)
#             i = (i + 1) % self.conf.LICZBA_ZRODEL
#
#     def wyslij_pszczoly_zwiadowcze(self):
#         if np.amax(self.niepowodzenia) >= self.conf.LIMIT:
#             i = self.niepowodzenia.argmax()
#             self.rozwiazania[i] = self.conf.DOLNA_GRANICA + np.random.rand(self.conf.WYMIAR) * (
#                         self.conf.GORNA_GRANICA - self.conf.DOLNA_GRANICA)
#             self.dopasowanie[i] = self.kalkulator_dopasowania(self.kalkulator_funkcji(self.rozwiazania[i]))
#             self.niepowodzenia[i] = 0
#
#     def zwieksz_liczbe_iteracji(self):
#         self.iteracje += 1
#
#     def warunek_stopu(self):
#
#         status = self.iteracje >= self.conf.MAKSYMALNA_LICZBA_ITERACJI
#
#         return status
#
#     def zapamietaj_najlepsze_rozwiazanie(self):
#         var = self.kalkulator_funkcji(self.rozwiazania[np.argmax(self.dopasowanie)])
#         if var < self.najlepszy:
#             self.naj_rozwiazanie = np.copy(self.rozwiazania[np.argmax(self.dopasowanie)])
#             self.najlepszy = var
#
#     def zwieksz_cykl(self):
#         self.cykl += 1
#
#
# def main(argv):
#     conf = Config(argv)
#     abc = AlgorytmABC(conf)
#     abc.inicjalizuj()
#     abc.zapamietaj_najlepsze_rozwiazanie()
#
#     while not abc.warunek_stopu():
#         abc.wyslij_pszczoly_zatrudnione()
#         abc.kalkulator_prawdopodobienstwa()
#         abc.wyslij_pszczoly_obserwujace()
#         abc.zapamietaj_najlepsze_rozwiazanie()
#         abc.wyslij_pszczoly_zwiadowcze()
#         abc.zwieksz_cykl()
#
#         print(f"{abc.cykl}: {abc.najlepszy}")
#
#     print("Najlepsze dopasowanie:", abc.najlepszy)
#     print("Najlepsze Rozwiązanie:", abc.naj_rozwiazanie)
#
#
# if __name__ == '__main__':
#     main(sys.argv[1:])


import numpy as np
import random
from typing import List, Type
import numpy as np
import random

# Import funkcji testowych
from .test_functions import *

# Jellyfish Search Algorithm
def jellyfish_search(objective_function, dimensions, lower_boundary, upper_boundary, population_size, max_iterations):
    jellyfish = np.random.uniform(lower_boundary, upper_boundary, (population_size, dimensions))
    fitness = np.array([objective_function(position) for position in jellyfish])

    best_jellyfish = jellyfish[np.argmin(fitness)]
    best_fitness = np.min(fitness)

    beta = 3.0
    gamma = 0.1

    for iteration in range(max_iterations):
        for current in range(population_size):
            time_control = abs((1 - iteration / max_iterations) * (2 * np.random.rand() - 1))

            if time_control >= 0.5:
                micro = np.mean(jellyfish, axis=0)
                ocean_current = best_jellyfish - beta * np.random.rand() * micro
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
                    best_jellyfish = jellyfish[current].copy()
                    best_fitness = new_fitness

    return best_jellyfish, best_fitness


def artificial_bee_colony(func, lower_boundary, upper_boundary, dimensions, population_size, max_iterations, limit):
    # Inicjalizacja parametrów
    population = np.random.uniform(lower_boundary, upper_boundary, (population_size, dimensions))
    fitness = np.array([1 / (1 + func(solution)) if func(solution) >= 0 else 1 + abs(func(solution)) for solution in population])
    failures = np.zeros(population_size)
    best_solution = population[np.argmax(fitness)]
    best_fitness = func(best_solution)

    def generate_new_solution(index):
        dimension = random.randint(0, dimensions - 1)
        neighbor = random.randint(0, population_size - 1)
        while neighbor == index:
            neighbor = random.randint(0, population_size - 1)

        new_solution = population[index].copy()
        diff = new_solution[dimension] - population[neighbor][dimension]
        new_solution[dimension] += (random.random() - 0.5) * 2 * diff
        new_solution = np.clip(new_solution, lower_boundary, upper_boundary)

        new_fitness = 1 / (1 + func(new_solution)) if func(new_solution) >= 0 else 1 + abs(func(new_solution))
        if new_fitness > fitness[index]:
            population[index] = new_solution
            fitness[index] = new_fitness
            failures[index] = 0
        else:
            failures[index] += 1

    # Algorytm ABC
    for iteration in range(max_iterations):
        # Faza pszczół zatrudnionych
        for i in range(population_size):
            generate_new_solution(i)

        # Faza pszczół naśladowczych
        probabilities = fitness / fitness.sum()
        for i in range(population_size):
            if random.random() < probabilities[i]:
                generate_new_solution(i)

        # Faza pszczół zwiadowczych
        for i in range(population_size):
            if failures[i] > limit:
                population[i] = np.random.uniform(lower_boundary, upper_boundary, dimensions)
                fitness[i] = 1 / (1 + func(population[i])) if func(population[i]) >= 0 else 1 + abs(func(population[i]))
                failures[i] = 0

        # Aktualizacja najlepszego rozwiązania
        current_best = population[np.argmax(fitness)]
        current_fitness = func(current_best)
        if current_fitness < best_fitness:
            best_solution = current_best
            best_fitness = current_fitness

    return best_solution, best_fitness


# Benchmarking
def run_benchmarks():
    benchmark_functions = [
        (rastrigin, rastrigin_properties()),
        (rosenbrock, rosenbrock_properties()),
        (sphere, sphere_properties()),
        (beale, beale_properties()),
        (bukin, bukin_properties()),
        (himmelblaus, himmelblaus_properties())
    ]

    results = []
    for func, properties in benchmark_functions:
        for dim in properties['dimensions']:
            best_jellyfish, fitness_jellyfish = jellyfish_search(
                func=func,
                dimensions=dim,
                lower_boundary=properties['lower_boundary'],
                upper_boundary=properties['upper_boundary'],
                population_size=30,
                max_iterations=100
            )

            best_abc, fitness_abc = artificial_bee_colony(
                func=func,
                lower_boundary=properties['lower_boundary'],
                upper_boundary=properties['upper_boundary'],
                dimensions=dim,
                population_size=30,
                max_iterations=100,
                limit=20
            )

            results.append({
                "function": properties['name'],
                "dimension": dim,
                "jellyfish_best_fitness": fitness_jellyfish,
                "abc_best_fitness": fitness_abc
            })
    return results