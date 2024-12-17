import json
import os


def save_state(algorithm_name, state):
    with open(f"{algorithm_name}_state.json", "w") as f:
        json.dump(state, f)


def load_state(algorithm_name):
    if os.path.exists(f"{algorithm_name}_state.json"):
        with open(f"{algorithm_name}_state.json", "r") as f:
            return json.load(f)
    return None
