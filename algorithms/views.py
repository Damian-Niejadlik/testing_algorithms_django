import threading
import time
import numpy as np
from django.shortcuts import render
from django.http import JsonResponse
from .algorithms import jellyfish_search, artificial_bee_colony
from .test_functions import rastrigin, rastrigin_properties  # Przykład z benchmarków
import threading

# Zmienna do przechowywania aktywnych wątków
current_threads = {}
algorithm_states = {}  # Przechowywanie stanu algorytmów (PAUSED, RUNNING, itp.)


def home(request):
    """Strona główna."""
    return render(request, 'home.html')


def sphere(x):
    return sum(xi ** 2 for xi in x)


# Helper funkcja do uruchamiania algorytmów
def run_algorithm_in_background(algorithm_name, func, params):
    """
    Funkcja uruchamia algorytm w tle, aktualizuje stan i postęp algorytmu.
    """
    global algorithm_states
    progress = 0
    algorithm_states[algorithm_name] = {'status': 'running', 'progress': progress, 'result': None}

    def algorithm_runner():
        nonlocal progress
        try:
            lower = params['lower_boundary']
            upper = params['upper_boundary']
            dimensions = params['dimensions']

            # Uruchomienie odpowiedniego algorytmu
            if algorithm_name == "jellyfish":
                result, fitness = jellyfish_search(func, dimensions, lower, upper, 30, 100)
            elif algorithm_name == "bee_colony":
                result, fitness = artificial_bee_colony(func, lower, upper, dimensions, 30, 100, 20)
            else:
                raise ValueError("Nieznany algorytm.")

            algorithm_states[algorithm_name]['status'] = 'completed'
            algorithm_states[algorithm_name]['result'] = {'solution': result.tolist(), 'fitness': fitness}
        except Exception as e:
            algorithm_states[algorithm_name]['status'] = 'error'
            algorithm_states[algorithm_name]['error'] = str(e)

    threading.Thread(target=algorithm_runner).start()


# Widok strony testowej
def algorithm_test(request, algorithm_name):
    """
    Renderuje stronę HTML do testowania algorytmów.
    """
    return render(request, 'algorithm_test.html', {'algorithm_name': algorithm_name})


# Rozpoczęcie algorytmu
def start_algorithm(request, algorithm_name):
    """
    Uruchamia wybrany algorytm.
    """
    if algorithm_states.get(algorithm_name, {}).get('status') in ['running', 'paused']:
        return JsonResponse({'status': 'already_running'})

    # Parametry dla algorytmów - można je dostosować dynamicznie
    func = rastrigin  # Przykład funkcji celu
    params = rastrigin_properties()
    params['dimensions'] = 10  # Ustawienie wymiaru dla testu

    run_algorithm_in_background(algorithm_name, func, params)
    return JsonResponse({'status': 'started'})


# Postęp algorytmu
def algorithm_progress(request, algorithm_name):
    """
    Zwraca status i postęp działania algorytmu.
    """
    state = algorithm_states.get(algorithm_name, {'status': 'not_started', 'progress': 0})
    return JsonResponse(state)

