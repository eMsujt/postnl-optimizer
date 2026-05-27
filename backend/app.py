"""
PostNL Route Optimizer — OR-Tools TSP backend
Deploy on Render (free or starter plan) and paste the URL into app.js.

v2: accepts an optional 'duration_matrix' field (n×n road-duration seconds from OSRM)
    so OR-Tools optimises by real road travel time instead of straight-line distance.
    Falls back to haversine when no matrix is supplied.
"""

import math
from flask import Flask, request, jsonify
from flask_cors import CORS
from ortools.constraint_solver import routing_enums_pb2, pywrapcp

app = Flask(__name__)
CORS(app)


def haversine_m(lat1, lng1, lat2, lng2):
    R = 6_371_000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return int(R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))


def build_haversine_matrix(coords):
    n = len(coords)
    return [
        [0 if i == j else haversine_m(coords[i][0], coords[i][1], coords[j][0], coords[j][1])
         for j in range(n)]
        for i in range(n)
    ]


def blend_matrix(stop_coords, depot_coords, road_matrix):
    """
    Build an (n+1)×(n+1) distance matrix:
      row/col 0       = depot — uses haversine for all depot arcs
      rows/cols 1..n  = delivery stops — uses OSRM road durations
    Null entries in road_matrix fall back to haversine.
    """
    n = len(stop_coords)
    all_coords = [depot_coords] + list(stop_coords)
    hav = build_haversine_matrix(all_coords)

    matrix = [[0] * (n + 1) for _ in range(n + 1)]
    # Depot row and column: always haversine
    for j in range(n + 1):
        matrix[0][j] = hav[0][j]
        matrix[j][0] = hav[j][0]
    # Stop-to-stop: prefer OSRM road durations, fall back to haversine
    for i in range(n):
        for j in range(n):
            v = road_matrix[i][j]
            matrix[i + 1][j + 1] = int(v) if (v is not None and v >= 0) else hav[i + 1][j + 1]

    return matrix


def solve_tsp(matrix, depot=0, time_limit=20):
    n = len(matrix)
    if n <= 1:
        return list(range(n))

    manager = pywrapcp.RoutingIndexManager(n, 1, depot)
    routing = pywrapcp.RoutingModel(manager)

    def dist_cb(from_idx, to_idx):
        return matrix[manager.IndexToNode(from_idx)][manager.IndexToNode(to_idx)]

    cb_idx = routing.RegisterTransitCallback(dist_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(cb_idx)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy    = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.seconds = time_limit

    solution = routing.SolveWithParameters(params)
    if not solution:
        return list(range(n))

    route, idx = [], routing.Start(0)
    while not routing.IsEnd(idx):
        route.append(manager.IndexToNode(idx))
        idx = solution.Value(routing.NextVar(idx))
    return route


@app.route('/health')
def health():
    return jsonify({'status': 'ok'})


@app.route('/optimize', methods=['POST'])
def optimize():
    data  = request.get_json(force=True)
    stops = data.get('stops', [])
    start = data.get('start_location')
    # Optional OSRM road-duration matrix: n×n seconds, stop indices only (no depot row)
    road_matrix = data.get('duration_matrix')

    if len(stops) < 2:
        return jsonify({'order': list(range(len(stops))), 'distance_m': 0})

    n = len(stops)
    stop_coords = [(s['lat'], s['lng']) for s in stops]

    # Validate road matrix dimensions — silently ignore malformed payloads
    use_road = (
        road_matrix is not None
        and isinstance(road_matrix, list)
        and len(road_matrix) == n
        and all(isinstance(row, list) and len(row) == n for row in road_matrix)
    )

    if start:
        depot = (start['lat'], start['lng'])

        if use_road:
            matrix = blend_matrix(stop_coords, depot, road_matrix)
        else:
            all_coords = [depot] + stop_coords
            matrix = build_haversine_matrix(all_coords)

        # Open TSP: driver does not need to return to depot → zero out return arcs
        for row in matrix:
            row[0] = 0

        raw   = solve_tsp(matrix, depot=0)
        order = [i - 1 for i in raw if i > 0]
        total = sum(matrix[raw[k]][raw[k + 1]] for k in range(len(raw) - 1))
    else:
        if use_road:
            matrix = [[int(road_matrix[i][j]) if road_matrix[i][j] is not None else 0
                       for j in range(n)] for i in range(n)]
        else:
            matrix = build_haversine_matrix(stop_coords)

        order = solve_tsp(matrix, depot=0)
        total = sum(matrix[order[k]][order[k + 1]] for k in range(len(order) - 1))

    return jsonify({'order': order, 'distance_m': total})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
