import json, sys
import numpy as np
import pandas as pd

from helper import *

import logging
from rich.logging import RichHandler
from rich.progress import track

# logging parameters
log = logging.getLogger('rich')
logging.basicConfig(level='NOTSET', format='%(message)s', datefmt='[%X]', handlers=[RichHandler()])

if __name__ == "__main__":

    data_folder = 'data'
    if len(sys.argv) > 1:
        data_folder = sys.argv[1]

    # loading build inputs
    ROUTE_DATA = f'{data_folder}/model_build_inputs/route_data.json'
    ACTUAL_SEQUENCES = f'{data_folder}/model_build_inputs/actual_sequences.json'
    TRAVEL_TIMES = f'{data_folder}/model_build_inputs/travel_times.json'

    log.info(f'GET /{TRAVEL_TIMES}')
    df_travel = pd.read_json(TRAVEL_TIMES)
    log.info(f'GET /{ROUTE_DATA}')
    df_route = pd.read_json(ROUTE_DATA)
    log.info(f'GET /{ACTUAL_SEQUENCES}')
    df_actual = pd.read_json(ACTUAL_SEQUENCES)

    df_route_T = df_route.T
    routes = df_route.columns
    stations = list(set(df_route_T['station_code']))

    ZONE_CENTERS = f'{data_folder}/model_build_outputs/zone_centers.json'
    log.info(f'POST /{ZONE_CENTERS}')
    zone_coords = {station: {station: []} for station in stations}
    zone_centers = {station: {station: []} for station in stations}

    for station in stations:

        # store all coordinates based on station and zone id
        routes_from_station = df_route[routes[(df_route_T['station_code'] == station)]].loc['stops']

        for route in routes_from_station:
            for stop in route:
                zone_id = route[stop]['zone_id']
                coords = (route[stop]['lat'], route[stop]['lng'])

                if route[stop]['type'] == 'Station':
                    zone_coords[station][station] += [coords]
                elif zone_id:
                    if zone_id not in zone_coords[station]:
                        zone_coords[station][zone_id] = [coords]
                    else:
                        zone_coords[station][zone_id] += [coords]

        # compute zone centers
        for zone_id, stop_coords in zone_coords[station].items():
            mean_lat, mean_lng = np.mean(stop_coords, axis=0)
            zone_centers[station][zone_id] = [mean_lat, mean_lng]

    with open(ZONE_CENTERS, 'w') as file:
        json.dump(zone_centers, file)

    log.info('Compute count matrices')
    # convert stop sequences to zone sequences
    zone_seqs_dict = {}

    for route_id in routes:
        C = df_travel[route_id].dropna()  # reading travel times
        zone_seqs_dict[route_id] = optimal_zone_seq(route_id, df_route, df_actual, C)

    # initialize count matrices for each station
    count_matrices = {}
    for station in stations:
        routes_from_station = df_route_T[df_route_T['station_code'] == station].index
        zone_seqs = [zone_seqs_dict[route_id] for route_id in routes_from_station]
        zones_from_station = [station] + list(set(zone for zone_seq in zone_seqs for zone in zone_seq))
        count_matrices[station] = pd.DataFrame(0, index=zones_from_station, columns=zones_from_station)

    # update count matrices based on zone sequences
    for route_id in routes:
        station = df_route[route_id]['station_code']
        zone_sequence = [station] + zone_seqs_dict[route_id]

        for i in range(len(zone_sequence)):
            count_matrices[station].loc[zone_sequence[i-1], zone_sequence[i]] += 1

    # compute and store probability matrices for each station
    log.info('POST count matrices')
    for station_index, station in track(enumerate(stations), total=len(stations)):
        count_matrix = count_matrices[station]
        # (count_matrix.T).to_csv(f'{data_folder}/model_build_outputs/counts_from_{station}.csv')
        prob_matrix = count_matrix.div(count_matrix.sum(axis=1), axis=0).T
        prob_matrix.to_csv(f'{data_folder}/model_build_outputs/probs_from_{station}.csv')

    log.info('Finished')
