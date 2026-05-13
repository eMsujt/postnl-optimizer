"""
PostNL Route Optimizer — OR-Tools TSP backend
Deploy on Render (free or starter plan) and paste the URL into app.js.
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


def build_matrix(coords):
    n = len(coords)
    return [
        [0 if i == j else haversine_m(coords[i][0], coords[i][1], coords[j][0], coords[j][1])
         for j in range(n)]
        for i in range(n)
    ]


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

    if len(stops) < 2:
        return jsonify({'order': list(range(len(stops))), 'distance_m': 0})

    stop_coords = [(s['lat'], s['lng']) for s in stops]

    if start:
        # Node 0 = driver's current GPS position (depot), nodes 1..n = delivery stops.
        # Open TSP: driver does not return to depot → zero out return arcs.
        all_coords = [(start['lat'], start['lng'])] + stop_coords
        matrix = build_matrix(all_coords)
        for row in matrix:
            row[0] = 0  # free to end anywhere
        raw = solve_tsp(matrix, depot=0)
        order = [i - 1 for i in raw if i > 0]
        total_dist = sum(matrix[raw[i]][raw[i + 1]] for i in range(len(raw) - 1))
    else:
        matrix = build_matrix(stop_coords)
        order = solve_tsp(matrix, depot=0)
        total_dist = sum(matrix[order[i]][order[i + 1]] for i in range(len(order) - 1))

    return jsonify({'order': order, 'distance_m': total_dist})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
