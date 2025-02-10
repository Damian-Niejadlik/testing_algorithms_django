import json
import csv
from urllib.parse import unquote
from django.http import JsonResponse, HttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
import openpyxl
from .algorithms import jellyfish_search, artificial_bee_colony, pause_algorithm, resume_algorithm
from .test_functions import (
    rastrigin, rastrigin_properties,
    rosenbrock, rosenbrock_properties,
    sphere, sphere_properties,
    beale, beale_properties,
    bukin, bukin_properties,
    himmelblaus, himmelblaus_properties
)

FUNCTIONS = {
    'Rastrigin': (rastrigin, rastrigin_properties),
    'Rosenbrock': (rosenbrock, rosenbrock_properties),
    'Sphere': (sphere, sphere_properties),
    'Beale': (beale, beale_properties),
    'Bukin': (bukin, bukin_properties),
    'Himmelblaus': (himmelblaus, himmelblaus_properties),
}

ALGORITHMS = {
    "Jellyfish Search": jellyfish_search,
    "Artificial Bee Colony": artificial_bee_colony,
}


@csrf_exempt
def pause_view(request):
    if request.method == "POST":
        pause_algorithm()
        request.session['paused'] = True
        return JsonResponse({"message": "Algorithm paused successfully."})
    return HttpResponse(status=400)

@csrf_exempt
def resume_view(request):
    if request.method == "POST":
        resume_algorithm()
        request.session['paused'] = False
        return JsonResponse({"message": "Algorithm resumed successfully."})
    return HttpResponse(status=400)


def home(request):
    algorithms = list(ALGORITHMS.keys())
    return render(request, "home.html", {"algorithms": algorithms})


def algorithm_test(request, algorithm_name):
    algorithm_name = unquote(algorithm_name)
    return render(request, "algorithm_test.html", {"algorithm_name": algorithm_name, "functions": FUNCTIONS})


import json
from urllib.parse import unquote
from django.http import JsonResponse, HttpResponse

def start_algorithm(request):
    global RESULTS
    request.session['paused'] = False

    if request.method == "POST":
        data = json.loads(request.body)
        algorithm_name = unquote(data["algorithm"])
        function_names = data.get("functions", [])  # Pobiera listę wybranych funkcji

        # Sprawdzenie, czy przekazano listę funkcji
        if not isinstance(function_names, list) or not function_names:
            return JsonResponse({"error": "At least one function must be selected."}, status=400)

        try:
            dimensions = int(data["dimensions"])
            population_size = int(data["population_size"])
            max_iterations = int(data["max_iterations"])
        except ValueError:
            return JsonResponse({"error": "Dimensions, population size, and max iterations must be integers."}, status=400)

        if algorithm_name not in ALGORITHMS:
            return JsonResponse({"error": "Invalid algorithm."}, status=400)

        results_list = []
        algorithm_function = ALGORITHMS[algorithm_name]

        for function_name in function_names:
            if function_name not in FUNCTIONS:
                continue  # Pomija błędne funkcje

            objective_function, properties = FUNCTIONS[function_name]
            lower_boundary = properties()['lower_boundary']
            upper_boundary = properties()['upper_boundary']

            try:
                best_solution, best_fitness = algorithm_function(
                    objective_function, dimensions, lower_boundary, upper_boundary, population_size, max_iterations
                )
            except TypeError as e:
                return JsonResponse({"error": f"Algorithm execution failed for {function_name}: {e}"}, status=500)

            results_list.append({
                "function_name": function_name,
                "best_solution": str(best_solution),
                "best_fitness": best_fitness,
            })

        request.session['results'] = {
            "algorithm_name": algorithm_name,
            "functions": function_names,
            "dimensions": dimensions,
            "population_size": population_size,
            "max_iterations": max_iterations,
            "results_list": results_list
        }

        RESULTS = request.session['results'].copy()

        return JsonResponse({
            "message": "Algorithm executed successfully.",
            "download_links": {
                "xlsx": "/download/results.xlsx",
                "csv": "/download/results.csv",
            },
            "results": results_list
        })

    return HttpResponse(status=400)



def result_view(request):
    global RESULTS
    results = request.session.get('results')
    algorithm_name = RESULTS.get("algorithm_name")

    if not results:
        return HttpResponse("No results available.", status=400)
    return render(request, "result.html", {"results": results, "algorithm_name": algorithm_name})


def download_file(request, file_format):
    results = request.session.get('results')

    if not results:
        return JsonResponse({
            'error': 'No results found. Please run the algorithm first.'
        }, status=400)

    if file_format not in ['xlsx', 'csv']:
        return JsonResponse({
            'error': 'Invalid format requested.'
        }, status=400)

    if file_format == 'xlsx':
        response = HttpResponse(
            content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
        response['Content-Disposition'] = 'attachment; filename="results.xlsx"'
        generate_excel_file(response, results)
        return response

    else:
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="results.csv"'
        generate_csv_file(response, results)
        return response


def generate_excel_file(response, results):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Results"

    headers = ["Algorithm name", "Function name", "Dimensions",
               "Population size", "Max iterations", "Best solution",
               "Best fitness"]
    ws.append(headers)

    # Pobieranie głównych ustawień
    algorithm_name = results.get('algorithm_name', '')
    dimensions = results.get('dimensions', '')
    population_size = results.get('population_size', '')
    max_iterations = results.get('max_iterations', '')

    results_list = results.get('results_list', [])

    # Jeśli lista wyników jest pusta, logujemy problem
    if not results_list:
        print("Brak funkcji testowych w results:", results)
        return

    # Iterujemy po każdej funkcji z osobna
    for result in results_list:
        ws.append([
            algorithm_name,
            result.get('function_name', ''),
            dimensions,
            population_size,
            max_iterations,
            result.get('best_solution', ''),
            result.get('best_fitness', '')
        ])

    wb.save(response)


def generate_csv_file(response, results):
    writer = csv.writer(response)

    headers = ["Algorithm name", "Function name", "Dimensions",
               "Population size", "Max iterations", "Best solution",
               "Best fitness"]
    writer.writerow(headers)

    algorithm_name = results.get('algorithm_name', '')
    dimensions = results.get('dimensions', '')
    population_size = results.get('population_size', '')
    max_iterations = results.get('max_iterations', '')

    results_list = results.get('results_list', [])

    if not results_list:
        print("Brak funkcji testowych w results:", results)
        return

    for result in results_list:
        writer.writerow([
            algorithm_name,
            result.get('function_name', ''),
            dimensions,
            population_size,
            max_iterations,
            result.get('best_solution', ''),
            result.get('best_fitness', '')
        ])