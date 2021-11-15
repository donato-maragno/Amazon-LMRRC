import json, sys, time
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances_argmin
from multiprocessing import Pool, cpu_count

from helper import *

import logging
from rich.logging import RichHandler
from rich.progress import track
# from sklearn.metrics.pairwise import euclidean_distances

# logging parameters
logging.disable(logging.DEBUG)
log = logging.getLogger('rich')
logging.basicConfig(level='NOTSET', format='%(message)s', datefmt='[%X]', handlers=[RichHandler()])


def predict_route(route_id):
    """
    Returns the proposed route for a given route id.
    """

    route_info = df_new_route[route_id]
    station_code = route_info['station_code']
    zone_centers_from_station = all_zone_dict_with_coord[station_code]
    stops = route_info['stops']

    for stop in stops:
        if stops[stop]['type'] == 'Station':
            station_name = stop

    all_zone_ids = all_zone_ids_dict[route_id]
    zone_ids_list = list(set(all_zone_ids))  # unique zone ids
    C = df_new_travel[route_id].dropna()  # reading travel times

    # getting set of nodes for zone sequence prediction
    stop_zone_map = {}
    stop_zone_map[station_name] = station_code  # add station as a zone

    # map zone to node closest to zone center
    for zone_id in zone_ids_list:
        nodes_list = []
        nodes_list_coords = []
        for stop in stops:
            if stops[stop]['zone_id'] == zone_id:
                nodes_list += [stop]
                nodes_list_coords += [(stops[stop]['lat'], stops[stop]['lng'])]

        zone_coords = [zone_centers_from_station[zone_id]]  # zone center
        index_node = pairwise_distances_argmin(zone_coords, nodes_list_coords, metric='euclidean')[0]  # closest to zone center
        stop_zone_map[nodes_list[index_node]] = zone_id

    prob_matrix = prob_matrices[station_code]

    # zones = [station_code] + zone_ids_list
    # zone_centers = [zone_centers_from_station[zone_id] for zone_id in zones]
    
    # D_first = pd.DataFrame(euclidean_distances(zone_centers), index=zones, columns=zones)
    T = {}  # travel times between closest nodes to zone centers
    P = {}  # probabilities of moving between zone centers

    for stop_from, zone_from in stop_zone_map.items():
        T[zone_from] = {zone_to: C[stop_from][stop_to] for stop_to, zone_to in stop_zone_map.items()} ## TODO
        # D[zone_from] = {zone_to: T[zone_from][zone_to] for zone_to in stop_zone_map.values()}

        # add row and column of zeroes if zone id not in history
        if zone_from not in prob_matrix.columns:
            prob_matrix.loc[zone_from] = 0
            prob_matrix[zone_from] = 0

    for zone_from in stop_zone_map.values():
        P[zone_from] = {zone_to: prob_matrix[zone_from][zone_to] for zone_to in stop_zone_map.values()}

    T = pd.DataFrame(T)
    P = pd.DataFrame(P)
    T = T / T.max()  # TODO: Check best normalization!!

    # create cost matrix
    # omega = 0.6
    # omega_start = omega
    # omega_end = omega

    O = T.copy()
    O.iloc[:, :] = omega
    O.iloc[:, 0] = omega_start
    O.iloc[0] = omega_end

    C = O * T + (1 - O) * (1 - P)
    # C = T / (O + (1 - O) * N)

    for i in range(len(C)):  # make diagonal zero
        C.iloc[i, i] = 0.0

    # optimizing on a zone level
    zone_list = list(stop_zone_map.values())  # zone list including station
    station_zone_seq = optimize_ZL(zone_list, station_code, C)

    if not station_zone_seq:  # use Google OR if PuLP too slow
        log.warning(f'Using Google OR for route {route_id}: zone sequence')
        station_zone_seq = optimize_GoogleOR_ZL(zone_list, station_code, C)

    ordered_zones = station_zone_seq[1:]

    # optimizing on a node level
    C = df_new_travel[route_id].dropna()
    ordered_nodes = [station_name]

    for zone_idx, zone in enumerate(ordered_zones):
        # get nodes within zone
        nodes = []
        for stop in stops:
            if stops[stop]['zone_id'] == zone:
                nodes += [stop]

        num_nodes = len(nodes)

        if num_nodes == 1:
            ordered_nodes += nodes
        elif num_nodes == 2:
            node_1, node_2 = nodes[0], nodes[1]
            prev_node = ordered_nodes[-1]

            # add closest node first
            if C[prev_node][node_1] < C[prev_node][node_2]:
                ordered_nodes += [node_1, node_2]
            else:
                ordered_nodes += [node_2, node_1]
        else:
            # solve OTSP with first and last dummy nodes
            if zone_idx == len(ordered_zones) - 1:
                last_node = station_name
            else:
                next_zone_id = ordered_zones[zone_idx + 1]
                last_node = [stop for stop, zone_id in stop_zone_map.items() if zone_id == next_zone_id][0]

            first_node = ordered_nodes[-1]
            nodes = [first_node] + nodes

            route, status_NL = optimize_NL(nodes, first_node, last_node, C)  # TODO: remove first node from nodes list??

            # check if dummy node is not between first node and last node
            if status_NL != 1 or route['dummy'] != first_node or route[last_node] != 'dummy':
                log.critical(f'\nPuLP failed: route {route_id}, zone {zone}')
                log.info(f'Using Google OR for nodes {nodes}')
                route_GOR = optimize_GoogleOR_NL(nodes, first_node, last_node, C)
                ordered_nodes += route_GOR[1:]
            else:
                for _ in range(len(route) - 3):
                    ordered_nodes += [route[ordered_nodes[-1]]]

    return {'proposed': {stop: index for index, stop in enumerate(ordered_nodes)}}


if __name__ == '__main__':

    start = time.time()
    if len(sys.argv) > 1:
        data_folder = sys.argv[1]
    if len(sys.argv) == 3:
        omega = float(sys.argv[2])
        omega_start = omega
        omega_end = omega
    elif len(sys.argv) == 5:
        omega = float(sys.argv[2])
        omega_start = float(sys.argv[3])
        omega_end = float(sys.argv[4])

    # loading apply inputs
    NEW_ROUTE_DATA = f'{data_folder}/model_apply_inputs/new_route_data.json'
    NEW_TRAVEL_TIMES = f'{data_folder}/model_apply_inputs/new_travel_times.json'

    log.info(f'GET /{NEW_ROUTE_DATA}')
    df_new_route = pd.read_json(NEW_ROUTE_DATA)
    log.info(f'GET /{NEW_TRAVEL_TIMES}')
    df_new_travel = pd.read_json(NEW_TRAVEL_TIMES)
    route_ids = df_new_route.columns

    # loading build outputs
    ZONE_CENTERS = f'{data_folder}/model_build_outputs/zone_centers.json'
    log.info(f'GET /{ZONE_CENTERS}')
    df_zone_centers = pd.read_json(ZONE_CENTERS)
    stations = list(set(df_new_route.loc['station_code']))

    # read and store probability matrices
    log.info('GET probability matrices')
    prob_matrices = {}

    for station in stations:
        PROBS_STATION = f'{data_folder}/model_build_outputs/probs_from_{station}.csv'
        try:
            prob_matrices[station] = pd.read_csv(PROBS_STATION, index_col=0)
        except:
            log.warning(f'No data available for station {station}')
            prob_matrices[station] = pd.DataFrame(0, columns=[station], index=[station])

    # replace Nones by closest zone_ids and store station coordinates
    station_coords = {}
    all_zone_ids_dict = {}
    for route_id in route_ids:

        station_code = df_new_route[route_id]['station_code']
        stops_dict = df_new_route[route_id]['stops']
        all_zone_ids = [stops_dict[stop]['zone_id'] for stop in stops_dict]
        stops = list(stops_dict.keys())
        C = df_new_travel[route_id].dropna()
        all_zone_ids = remove_Nones(station_code, all_zone_ids, stops_dict, stops, C)[0]

        for i, stop in enumerate(stops):
            stop_info = stops_dict[stop]
            if stop_info['type'] == 'Dropoff':
                stop_info['zone_id'] = all_zone_ids[i]
            elif station_code not in station_coords:  # add station coords if station is new
                station_coords[station_code] = (stop_info['lat'], stop_info['lng'])

        all_zone_ids_dict[route_id] = all_zone_ids

    # for each station, check if zones are in training data, if not, add center
    log.info('Compute zone centers')
    all_zone_dict_with_coord = {}  # store dict of all (un)visited zones

    for station in stations:
        if station in df_zone_centers:
            zone_ids_centers = df_zone_centers[station].dropna()
        else:
            log.info(f'--> Station {station} has never been seen')
            zone_ids_centers = pd.Series({station: station_coords[station]})

        zone_ids = list(zone_ids_centers.index)
        zone_centers = list(zone_ids_centers.values)

        unvisited_zones = {}
        routes_from_station = df_new_route[route_ids[(df_new_route.loc['station_code'] == station)]].loc['stops']

        for route in routes_from_station:
            for stop in route.keys():
                # check if zone was not visited before
                zone_candidate = route[stop]['zone_id']
                if zone_candidate and zone_candidate not in zone_ids:
                    coords = (route[stop]['lat'], route[stop]['lng'])

                    # store all coordinates for this unvisited zone
                    if zone_candidate not in unvisited_zones:
                        unvisited_zones[zone_candidate] = [coords]
                    else:
                        unvisited_zones[zone_candidate] += [coords]

        new_zone_ids = list(unvisited_zones.keys())
        new_zone_centers = []

        # compute unvisited zone centers
        for stop_coords in unvisited_zones.values():
            mean_lat, mean_lng = np.mean(stop_coords, axis=0)
            new_zone_centers += [(mean_lat, mean_lng)]

        all_labels = zone_ids + new_zone_ids
        all_centers = zone_centers + new_zone_centers
        all_zone_dict_with_coord[station] = dict(zip(all_labels, all_centers))

    # predicting the route sequences in parallel
    log.info(f'Predict routes (omega = {omega_start, omega, omega_end})')
    with Pool(processes=cpu_count()) as p:
        results = list(track(p.imap(predict_route, route_ids), total=len(route_ids)))

    # if omega_start == omega == omega_end:
        # PROPOSED_SEQUENCES = f'{data_folder}/model_apply_outputs/proposed_sequences_omega_{omega:.2f}.json'
    # else:
    PROPOSED_SEQUENCES = f'{data_folder}/model_apply_outputs/proposed_sequences_omegas_{omega_start:.2f}-{omega:.2f}-{omega_end:.2f}.json'

    log.info(f'POST /{PROPOSED_SEQUENCES}')
    with open(PROPOSED_SEQUENCES, 'w') as file:
        json.dump(dict(zip(route_ids, results)), file)

    log.info(f'Finished in {time.time() - start} seconds')
