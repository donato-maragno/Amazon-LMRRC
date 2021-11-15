import numpy as np
import pandas as pd
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2
import pulp
from scipy.spatial import distance
from itertools import groupby


def find_closest_node(start_node, nodes, C): # node level
    """
    Returns the node closest to start_node out of the
    list of nodes. The matrix C contains the travel
    times between the nodes.
    """
    return nodes[np.argmin([C[start_node][node] for node in nodes])]

def remove_Nones(station_code, all_zone_ids, stops_info, stop_sequence, C):
    """
    Given a list of zone_ids, replace all None values
    of zone_id by the closest zone_id within the
    route (preprocessing data).
    """

    n = len(all_zone_ids)

    for ith_stop in range(n):
        if not all_zone_ids[ith_stop]:  # if zone id is missing
            ith_stop_id = stop_sequence[ith_stop]

            candidate_nodes = []

            for i, node in enumerate(stop_sequence):
                if i != ith_stop and stops_info[node]['zone_id']:  # make sure candidates have zone ids
                    candidate_nodes += [node]

            closest_node = find_closest_node(ith_stop_id, candidate_nodes, C)
            all_zone_ids[ith_stop] = all_zone_ids[stop_sequence.index(closest_node)]

    station_zone_ids = [f'{station_code}_{zone_id}' for zone_id in all_zone_ids]

    return all_zone_ids, station_zone_ids

def optimal_zone_seq(route_id, route_data, sequence_data, C):
    """
    Given the route data and actual sequence, returns the
    corresponding optimal sequence of visited zone_ids.
    """

    station_code = route_data[route_id].iloc[0]

    # store actual/driven sequence of stops and corresponding zone_ids
    stops_info = route_data[route_id]['stops']
    actual_stops = sequence_data[route_id]['actual']
    actual_stop_seq = sorted(actual_stops, key=actual_stops.get)[1:]

    # store all zones within the route, remove None values
    all_zone_ids = [stops_info[stop]['zone_id'] for stop in actual_stop_seq]
    all_zone_ids = remove_Nones(station_code, all_zone_ids, stops_info, actual_stop_seq, C)[0]

    # getting list of consecutive runs in the sequence
    runs_list = [(zone_id, len(list(zone_count))) for zone_id, zone_count in groupby(all_zone_ids)]
    unique_zones, _ = zip(*runs_list)
    zone_max_count_ind = {}

    for zone in set(unique_zones):
        list_ind_count = [(ind, count) for ind, (zone_id, count) in enumerate(runs_list) if zone_id == zone]
        indices, counts = zip(*list_ind_count)
        zone_max_count_ind[zone] = indices[counts.index(max(counts))]

    zone_seq, _ = zip(*sorted(zone_max_count_ind.items(), key=lambda item: item[1]))

    return list(zone_seq)

def optimal_zone_sequence(route_id, route_data, sequence_data, C):  # TODO: fix bug with print(zone_list_dict['RouteID_0a8650ca-77d1-459b-a8dd-da8442bbf1cf'])
    """
    Given the route data and actual sequence, returns the
    corresponding optimal sequence of visited zone_ids,
    together with the number of visits within each zone_id.
    """

    station_code = route_data[route_id].iloc[0]

    # store actual/driven sequence of stops and corresponding zone_ids
    stops_info = route_data[route_id]['stops']
    actual_stops = sequence_data[route_id]['actual']
    actual_stop_seq = sorted(actual_stops, key=actual_stops.get)[1:]

    # store all zones within the route, remove None values
    all_zone_ids = [stops_info[stop]['zone_id'] for stop in actual_stop_seq]
    all_zone_ids, station_zone_id = remove_Nones(station_code, all_zone_ids, stops_info, actual_stop_seq, C)

    # merge stops with the same zone_id after each other into one zone_id
    prev_zone_id = ''  # last visited zone_id
    zone_id_count = 0  # counter for number of visits of the same zone_id
    zone_id_list = []  # sequence of zone_ids
    zone_id_count_list = []  # corresponding number of visits

    for zone_id in all_zone_ids:
        if zone_id != prev_zone_id:
            if zone_id_count:  # skip this for the first zone_id
                zone_id_count_list += [zone_id_count]  # store #visits of previous zone_id

            zone_id_list += [zone_id]  # store current/new zone_id
            zone_id_count = 1  # reset counter
        else:
            zone_id_count += 1

        prev_zone_id = zone_id

    zone_id_count_list += [zone_id_count]  # add #visits of last zone_id

    # remove duplicate zone_ids by keeping them at the most #visits
    for zone_id in zone_id_list:
        if zone_id_list.count(zone_id) > 1:  # there are duplicates

            # store all actual indices and counts corresponding to the zone_id
            dupl_indices = [i for i, zone_i in enumerate(zone_id_list) if zone_i == zone_id]
            dupl_counts = [zone_id_count_list[dupl_index] for dupl_index in dupl_indices]

            # add other visits to corresponding maximum #visits of the zone_id
            arg_max_ind = dupl_counts.index(max(dupl_counts))
            max_ind = dupl_indices[arg_max_ind]
            zone_id_count_list[max_ind] = sum([zone_id_count_list[i] for i in dupl_indices])

            # delete duplicate zone_ids and corresponding counts
            dupl_indices.pop(arg_max_ind)  # keep zone_id with highest #visits
            for dupl_index in dupl_indices[::-1]:  # make sure the indexing of popping goes well
                zone_id_list.pop(dupl_index)
                zone_id_count_list.pop(dupl_index)

    zones_dict = {}
    for id in set(station_zone_id):
        index_list = [i for i, e in enumerate(station_zone_id) if e == id]
        list_stops_dict = [actual_stop_seq[i] for i in index_list]
        zones_dict[id] = list_stops_dict

    return zone_id_list, zones_dict  # TODO: no route_id, zone_id_count_list?


def print_solution(manager, routing, solution):
    """Prints solution on console."""
    index = routing.Start(0)
    plan_output = 'Route for vehicle 0:\n'
    route_distance = 0
    tour = [0]
    while not routing.IsEnd(index):
        plan_output += ' {} ->'.format(manager.IndexToNode(index))
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        tour.append(index)
        route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
    plan_output += ' {}\n'.format(manager.IndexToNode(index))
    plan_output += 'Route distance: {}miles\n'.format(route_distance)

    return tour


def optimize_GoogleOR_ZL(nodes, first_node, C):
    """
    ### Let's pray we don't need this.
    """
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return C[from_node][to_node]

    nodes2 = nodes.copy()
    nodes2.pop(nodes2.index(first_node))
    nodes = [first_node] + nodes2

    manager = pywrapcp.RoutingIndexManager(len(nodes), 1, 0)
    routing = pywrapcp.RoutingModel(manager)
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC)
    solution = routing.SolveWithParameters(search_parameters)

    tour = []
    if solution:
        tour_ = print_solution(manager, routing, solution)

    tour = [nodes[i] for i in tour_[:-1]]
    return tour


def compute_matrix_withDummy(nodes, first_node, last_node, C_series):

    def create_df_int_index(C_matrix, nodes):
        mapping_dict_toNum = {}
        mapping_dict_toString = {}
        cols_list = list(C_matrix.index)
        for i in range(len(cols_list)):
            element = cols_list[i]
            mapping_dict_toNum[element] = i
            mapping_dict_toString[i] = element
        new_df = C_matrix.copy()
        new_df.columns = new_df.columns.to_series().map(mapping_dict_toNum)
        new_df.index = new_df.index.to_series().map(mapping_dict_toNum)  # converting
        nodes_series = pd.Series(nodes).map(mapping_dict_toNum)
        return new_df, nodes_series, mapping_dict_toString

    big_M = 1e10
    nodes_except_dummy = nodes+[last_node]
    nodes = nodes_except_dummy+['dummy']
    nodes2 = nodes.copy()
    nodes2.pop(nodes2.index(first_node))
    nodes = [first_node] + nodes2
    C_df_zone = pd.DataFrame(list(C_series), index=C_series.keys()).loc[nodes, nodes]
    C_dict = C_df_zone.to_dict()
    C_dict['dummy'] = {}

    for node in nodes:
        C_dict[node]['dummy'] = big_M
        C_dict['dummy'][node] = big_M
    C_dict['dummy'][first_node] = big_M
    C_dict[last_node]['dummy'] = big_M
    C_dict[first_node]['dummy'] = 0
    C_dict['dummy'][last_node] = 0
    C_dict['dummy']['dummy'] = 0
    C_df_new = pd.DataFrame(C_dict)
    C, nodes, mapping_dict_toString = create_df_int_index(C_df_new, nodes)
    return C, nodes, mapping_dict_toString


def optimize_GoogleOR_NL(nodes, first_node, last_node, C_dict):
    """
    ### Let's pray we don't need this as well.
    """

    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return C.loc[from_node, to_node]

    C, nodes, mapping_dict_toString = compute_matrix_withDummy(nodes, first_node, last_node, C_dict)

    manager = pywrapcp.RoutingIndexManager(len(nodes), 1, 0)
    routing = pywrapcp.RoutingModel(manager)
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC)
    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        tour_ = print_solution(manager, routing, solution)
    tour = [nodes[i] for i in tour_[:-3]]
    return list(pd.Series(tour).map(mapping_dict_toString))


def optimize_ZL(nodes, first_node, C):
    """
    Finds the optimal sequence of nodes
    given the set of nodes, the first
    node and the expected travel times.
    """

    model = pulp.LpProblem('CVRP', pulp.LpMinimize)
    N = [node for node in nodes if node != first_node]
    u = pulp.LpVariable.dicts('u', (i for i in nodes), cat='Continuous', lowBound=0)
    x = pulp.LpVariable.dicts('x', ((i, j) for i in nodes for j in nodes), cat='Binary')

    # objective function
    model += sum(C[i][j] * x[i, j] for i in nodes for j in nodes)

    # flow constraints
    for node_i in nodes:
        model += sum(x[node_i, j] for j in nodes if node_i != j) == 1
    for node_j in nodes:
        model += sum(x[i, node_j] for i in nodes if node_j != i) == 1

    # Wikipedia source
    for j in N:
        for i in N:
            if i != j:
                model += u[i] - u[j] + x[i, j] * len(nodes) <= len(N)
    for i in N:
        model += (1 <= u[i] <= len(N))

    status = model.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=30))

    if status != 1:
        return None

    tour = dict([(i, j) for i in nodes for j in nodes if pulp.value(x[i, j]) == 1])

    ordered_nodes = [first_node]
    for i in range(len(tour) - 1):
        ordered_nodes += [tour[ordered_nodes[-1]]]

    return ordered_nodes


def optimize_NL(nodes, first_node, last_node, C_):
    """
    Finds the optimal sequence of nodes
    given the set of nodes, the first
    node and the expected travel times.
    """
    big_M = 1e4
    nodes_except_dummy = nodes+[last_node]
    nodes = nodes_except_dummy+['dummy']
    model = pulp.LpProblem('CVRP', pulp.LpMinimize)
    N = [node for node in nodes if node not in first_node]
    u = pulp.LpVariable.dicts('u', (i for i in nodes), cat='Continuous', lowBound=0)
    x = pulp.LpVariable.dicts('x', ((i, j) for i in nodes for j in nodes), cat='Binary')

    # This is use to be sure that the choice of i in x[i, station_node(first_node)]==1
    # is not based on the cost between i and station_node
    C_['dummy'] = {}
    for node in N:
        C_['dummy'][node] = big_M
        C_[node]['dummy'] = big_M
    C_[first_node]['dummy'] = big_M
    C_[first_node][last_node] = big_M
    C_['dummy'][last_node] = big_M
    C_['dummy'][first_node] = 0
    C_[last_node]['dummy'] = 0
    C_['dummy']['dummy'] = 0

    # objective function
    model += sum(C_[i][j] * x[i, j] for i in nodes for j in nodes)

    # flow constraints
    for node_i in nodes:
        model += sum(x[node_i, j] for j in nodes if node_i != j) == 1
    for node_j in nodes:
        model += sum(x[i, node_j] for i in nodes if node_j != i) == 1

    # Wikipedia source
    for j in N:
        for i in N:
            if i != j:
                model += u[i] - u[j] + x[i, j] * len(nodes) <= len(N)
    for i in N:
        model += (1 <= u[i] <= len(N))

    status = model.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=30))

    if status != 1:
        return 'NOTHING', status

    tour = dict([(i, j) for i in nodes for j in nodes if pulp.value(x[i, j]) == 1])
    return tour, status
