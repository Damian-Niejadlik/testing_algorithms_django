import json
import csv
from urllib.parse import unquote
from django.http import JsonResponse, HttpResponse
from django.shortcuts import render
import openpyxl
from .algorithms import jellyfish_search, artificial_bee_colony, benchmark_algorithm
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


def home(request):
    algorithms = list(ALGORITHMS.keys())
    return render(request, "home.html", {"algorithms": algorithms})


def algorithm_test(request, algorithm_name):
    algorithm_name = unquote(algorithm_name)  # Decode URL parameter
    return render(request, "algorithm_test.html", {"algorithm_name": algorithm_name, "functions": FUNCTIONS})


def start_algorithm(request):
    global RESULTS
    if request.method == "POST":
        data = json.loads(request.body)
        algorithm_name = unquote(data["algorithm"])
        function_name = data["function"]

        try:
            dimensions = int(data["dimensions"])
            population_size = int(data["population_size"])
            max_iterations = int(data["max_iterations"])
        except ValueError:
            return JsonResponse({"error": "Dimensions, population size, and max iterations must be integers."},
                                status=400)

        if algorithm_name not in ALGORITHMS or function_name not in FUNCTIONS:
            return JsonResponse({"error": "Invalid algorithm or function."}, status=400)

        objective_function, properties = FUNCTIONS[function_name]
        lower_boundary = properties()['lower_boundary']
        upper_boundary = properties()['upper_boundary']

        algorithm_function = ALGORITHMS[algorithm_name]
        try:
            best_solution, best_fitness = algorithm_function(
                objective_function, dimensions, lower_boundary, upper_boundary, population_size, max_iterations
            )
        except TypeError as e:
            return JsonResponse({"error": f"Algorithm execution failed: {e}"}, status=500)

        request.session['results'] = {
            "algorithm_name": algorithm_name,
            "function_name": function_name,
            "dimensions": dimensions,
            "population_size": population_size,
            "max_iterations": max_iterations,
            "best_solution": str(best_solution),
            "best_fitness": best_fitness,
        }

        RESULTS = request.session['results'].copy()

        return JsonResponse({
            "message": "Algorithm executed successfully.",
            "download_links": {
                "xlsx": "/download/results.xlsx",
                "csv": "/download/results.csv",
            }
        })
    return HttpResponse(status=400)


# def download_file(request, file_format):
#     if file_format not in ['xlsx', 'csv']:
#         return HttpResponse("Invalid format requested.", status=400)
#
#     if file_format == 'xlsx':
#         response = HttpResponse(content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
#         response['Content-Disposition'] = 'attachment; filename="results.xlsx"'
#         generate_excel_file(response)
#         return response
#
#     elif file_format == 'csv':
#         response = HttpResponse(content_type='text/csv')
#         response['Content-Disposition'] = 'attachment; filename="results.csv"'
#         generate_csv_file(response)
#         return response


# def generate_excel_file(response):
#     global RESULTS
#     wb = openpyxl.Workbook()
#     ws = wb.active
#     ws.title = "Results"
#     ws.append(["Algorithm name", "Function name", "Dimensions", "Population size", "Max iterations", "Bestsolution",
#                "Best fitness"])
#     ws.append([values for values in RESULTS.values()])
#     wb.save(response)
#
#
# def generate_csv_file(response):
#     global RESULTS
#     writer = csv.writer(response)
#     writer.writerow(
#         ["Algorithm name", "Function name", "Dimensions", "Population size", "Max iterations", "Bestsolution",
#          "Best fitness"])  # Nagłówki
#     writer.writerow([values for values in RESULTS.values()])


def result_view(request):
    global RESULTS
    results = request.session.get('results')
    algorithm_name = RESULTS.get("algorithm_name")
    if not results:
        return HttpResponse("No results available.", status=400)
    return render(request, "result.html", {"results": results, "algorithm_name": algorithm_name})


def download_file(request, file_format):
    # Get results from the session instead of global variable
    global RESULTS

    # Check if we have results to work with
    print(RESULTS)
    if not RESULTS:
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
        generate_excel_file(response, RESULTS)
        return response

    else:  # csv case
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="results.csv"'
        generate_csv_file(response, RESULTS)
        return response


def generate_excel_file(response, results):
    # Pass results as parameter instead of using global
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Results"

    # Define headers
    headers = ["Algorithm name", "Function name", "Dimensions",
               "Population size", "Max iterations", "Best solution",
               "Best fitness"]
    ws.append(headers)

    # Add the results row
    ws.append([values for values in results.values()])

    wb.save(response)


def generate_csv_file(response, results):
    # Pass results as parameter instead of using global
    writer = csv.writer(response)

    # Define headers
    headers = ["Algorithm name", "Function name", "Dimensions",
               "Population size", "Max iterations", "Best solution",
               "Best fitness"]
    writer.writerow(headers)

    # Add the results row
    writer.writerow([values for values in results.values()])
