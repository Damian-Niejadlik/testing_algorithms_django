import os
import json
from django.conf import settings
from django.http import JsonResponse, FileResponse, HttpResponse
from django.shortcuts import render
from openpyxl import Workbook
from .algorithms import jellyfish_search, artificial_bee_colony

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
    return render(request, "algorithm_test.html", {"algorithm_name": algorithm_name, "functions": FUNCTIONS})


def start_algorithm(request):
    if request.method == "POST":
        data = json.loads(request.body)
        algorithm_name = data["algorithm"]
        function_name = data["function"]
        dimensions = int(data["dimensions"])
        population_size = int(data["population_size"])
        max_iterations = int(data["max_iterations"])

        if algorithm_name not in ALGORITHMS:
            return JsonResponse({"error": "Algorithm not found."}, status=400)

        if function_name not in FUNCTIONS:
            return JsonResponse({"error": "Function not found."}, status=400)

        objective_function, properties = FUNCTIONS[function_name]
        lower_boundary = properties()['lower_boundary']
        upper_boundary = properties()['upper_boundary']

        algorithm_function = ALGORITHMS[algorithm_name]

        if algorithm_name == "Jellyfish Search" or "Jellyfish%20Search":
            best_solution, best_fitness = algorithm_function(
                objective_function, dimensions, lower_boundary, upper_boundary, population_size, max_iterations
            )
        elif algorithm_name == "Artificial Bee Colony" or "Artificial%20Bee%20Colony":
            best_solution, best_fitness = algorithm_function(
                objective_function, lower_boundary, upper_boundary, dimensions, population_size, max_iterations
            )
        else:
            return JsonResponse({"error": "Algorithm implementation is missing."}, status=500)






        results_file_cls = f'results_{algorithm_name.lower().replace(" ", "_")}_{function_name.lower()}.cls'
        results_path_cls = os.path.join(settings.MEDIA_ROOT, results_file_cls)
        with open(results_path_cls, "w") as f:
            f.write("Algorithm Results\n")
            f.write(f"Algorithm: {algorithm_name}\n")
            f.write(f"Function: {function_name}\n")
            f.write(f"Dimensions: {dimensions}\n")
            f.write(f"Population Size: {population_size}\n")
            f.write(f"Max Iterations: {max_iterations}\n")
            f.write(f"Best Solution: {best_solution}\n")
            f.write(f"Best Fitness: {best_fitness}\n")

        results_file_xlsx = f'results_{algorithm_name.lower().replace(" ", "_")}_{function_name.lower()}.xlsx'
        results_path_xlsx = os.path.join(settings.MEDIA_ROOT, results_file_xlsx)
        wb = Workbook()
        ws = wb.active
        ws.title = "Results"
        ws.append(["Algorithm", "Function", "Dimensions", "Population Size", "Max Iterations", "Best Solution", "Best Fitness"])
        ws.append([algorithm_name, function_name, dimensions, population_size, max_iterations, str(best_solution), best_fitness])
        wb.save(results_path_xlsx)

        return JsonResponse({
            "message": "Algorithm executed successfully.",
            "results_file_cls": results_file_cls,
            "results_file_xlsx": results_file_xlsx,
            "benchmark_output": os.path.join("benchmarks", f"benchmark_{algorithm_name.lower().replace(' ', '_')}.csv")
        })

    return HttpResponse(status=400)


def download_results(request, filename):
    file_path = os.path.join(settings.MEDIA_ROOT, filename)
    if os.path.exists(file_path):
        response = FileResponse(open(file_path, "rb"), content_type="application/octet-stream")
        response["Content-Disposition"] = f'attachment; filename="{filename}"'
        return response
    else:
        return HttpResponse("File not found.", status=404)
