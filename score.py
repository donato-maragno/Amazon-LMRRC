import numpy as np
import pandas as pd
import json
import sys


def read_json_data(filepath):
    '''
    Loads JSON file and generates a dictionary from it.

    Parameters
    ----------
    filepath : str
        Path of desired file.

    Raises
    ------
    JSONDecodeError
        The file exists and is readable, but it does not have the proper
        formatting for its place in the inputs of evaluate.

    Returns
    -------
    file : dict
        Dictionary form of the JSON file to which filepath points.

    '''
    try:
        with open(filepath, newline = '') as in_file:
            file=json.load(in_file)
            in_file.close()
    except FileNotFoundError:
        print("The '{}' file is missing!".format(filepath))
        sys.exit()
    except Exception as e:
        print("Error when reading the '{}' file!".format(filepath))
        print(e)
        sys.exit()
    return file

def good_format(file,input_type,filepath):
    '''
    Checks if input dictionary has proper formatting.
    
    Parameters
    ----------
    file : dict
        Dictionary loaded from evaluate input file.
    input_type : str
        Indicates which input of evaluate the current file is. Can be
        "actual," "proposed," "costs," or "invalids."
    filepath : str
        Path from which file was loaded.

    Raises
    ------
    JSONDecodeError
        The file exists and is readable, but it does not have the proper
        formatting for its place in the inputs of evaluate.

    Returns
    -------
    None.

    '''
    
    for route in file:
        if route[:8]!='RouteID_':
            raise JSONDecodeError('Improper route ID in {}. Every route must be denoted by a string that begins with "RouteID_".'.format(filepath))
    if input_type=='proposed' or input_type=='actual':
        for route in file:
            if type(file[route])!=dict or len(file[route])!=1: 
                raise JSONDecodeError('Improper route in {}. Each route ID must map to a dictionary with a single key.'.format(filepath))
            if input_type not in file[route]:
                if input_type=='proposed':
                    raise JSONDecodeError('Improper route in {}. Each route\'s dictionary in a proposed sequence file must have the key, "proposed".'.format(filepath))
                else:
                    raise JSONDecodeError('Improper route in {}. Each route\'s dictionary in an actual sequence file must have the key, "actual".'.format(filepath))
            if type(file[route][input_type])!=dict:
                raise JSONDecodeError('Improper route in {}. Each sequence must be in the form of a dictionary.'.format(filepath))
            num_stops=len(file[route][input_type])
            for stop in file[route][input_type]:
                if type(stop)!=str or len(stop)!=2:
                    raise JSONDecodeError('Improper stop ID in {}. Each stop must be denoted by a two-letter ID string.'.format(filepath))
                stop_num=file[route][input_type][stop]
                if type(stop_num)!=int or stop_num>=num_stops:
                    file[route][input_type][stop]='invalid'
    if input_type=='costs':
        for route in file:
            if type(file[route])!=dict:
                raise JSONDecodeError('Improper matrix in {}. Each cost matrix must be a dictionary.'.format(filepath)) 
            for origin in file[route]:
                if type(origin)!=str or len(origin)!=2:
                    raise JSONDecodeError('Improper stop ID in {}. Each stop must be denoted by a two-letter ID string.'.format(filepath))
                if type(file[route][origin])!=dict:
                    raise JSONDecodeError('Improper matrix in {}. Each origin in a cost matrix must map to a dictionary of destinations'.format(filepath))
                for dest in file[route][origin]:
                    if type(dest)!=str or len(dest)!=2:
                        raise JSONDecodeError('Improper stop ID in {}. Each stop must be denoted by a two-letter ID string.'.format(filepath))
                    if not(type(file[route][origin][dest])==float or type(file[route][origin][dest])==int):
                        raise JSONDecodeError('Improper time in {}. Every travel time must be a float or int.'.format(filepath))
    if input_type=='invalids':
        for route in file:
            if not(type(file[route])==float or type(file[route])==int):
                raise JSONDecodeError('Improper score in {}. Every score in an invalid score file must be a float or int.'.format(filepath))

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

def seq_dev_zone(actual,sub):
    '''
    Calculates sequence deviation.

    Parameters
    ----------
    actual : list
        Actual route.
    sub : list
        Submitted route.

    Returns
    -------
    float
        Sequence deviation.

    '''
    # actual=actual[1:-1]
    # sub=sub[1:-1]
    comp_list=[]
    for i in sub:
        comp_list.append(actual.index(i))
        comp_sum=0
    for ind in range(1,len(comp_list)):
        comp_sum+=abs(comp_list[ind]-comp_list[ind-1])-1
    n=len(actual)
    return (2/(n*(n-1)))*comp_sum

def optimal_zone_sequence_scr(route_id, route_data, sequence_data, travel_data, specify):
    """
    Given the route data and actual sequence, returns the
    corresponding optimal sequence of visited zone_ids,
    together with the number of visits within each zone_id.
    """
    
    station_code = route_data[route_id].iloc[0]
    total_travel_time = 0
    
    # store actual/driven sequence of stops and corresponding zone_ids
    stops_info = route_data[route_id]['stops']
    actual_stops = sequence_data[route_id][specify]
    actual_stop_sequence = sorted(actual_stops, key=actual_stops.get)[1:]

    # store all zones within the route, remove None values
    C = travel_data[route_id].dropna()  # reading travel times
    all_zone_ids = [stops_info[stop]['zone_id'] for stop in actual_stop_sequence]
    all_zone_ids, _ = remove_Nones(station_code, all_zone_ids, stops_info, actual_stop_sequence, C)
    
    for index, stop in enumerate(actual_stop_sequence):
        prev_stop = actual_stop_sequence[index-1]
        total_travel_time += C[prev_stop][stop]
    
    # merge stops with the same zone_id after each other into one zone_id
    prev_zone_id = '' # last visited zone_id
    zone_id_count = 0 # counter for number of visits of the same zone_id
    zone_id_list = [] # sequence of zone_ids
    zone_id_count_list = [] # corresponding number of visits
    
    for zone_id in all_zone_ids:
        if zone_id != prev_zone_id:
            if zone_id_count: # skip this for the first zone_id
                zone_id_count_list += [zone_id_count] # store #visits of previous zone_id
            
            zone_id_list += [zone_id] # store current/new zone_id
            zone_id_count = 1 # reset counter
        else:
            zone_id_count += 1
            
        prev_zone_id = zone_id

    zone_id_count_list += [zone_id_count] # add #visits of last zone_id
    
    # remove duplicate zone_ids by keeping them at the most #visits
    m = len(zone_id_list)
    
    for zone_id in zone_id_list:
        if zone_id_list.count(zone_id) > 1: # there are duplicates
            
            # store all actual indices and counts corresponding to the zone_id
            dupl_indices = [i for i, zone_i in enumerate(zone_id_list) if zone_i == zone_id]
            dupl_counts = [zone_id_count_list[dupl_index] for dupl_index in dupl_indices]
            
            # add other visits to corresponding maximum #visits of the zone_id
            arg_max_ind = dupl_counts.index(max(dupl_counts))
            max_ind = dupl_indices[arg_max_ind]
            zone_id_count_list[max_ind] = sum([zone_id_count_list[i] for i in dupl_indices])
            
            # delete duplicate zone_ids and corresponding counts
            dupl_indices.pop(arg_max_ind) # keep zone_id with highest #visits
            for dupl_index in dupl_indices[::-1]: # make sure the indexing of popping goes well
                zone_id_list.pop(dupl_index)
                zone_id_count_list.pop(dupl_index)
    
    return zone_id_list, total_travel_time


class JSONDecodeError(Exception):
    pass

def evaluate(df_new_route, actual_routes_json, submission_json, cost_matrices_json, invalid_scores_json,**kwargs):
    '''
    Calculates score for a submission.

    Parameters
    ----------
    actual_routes_json : str
        filepath of JSON of actual routes.
    submission_json : str
        filepath of JSON of participant-created routes.
    cost_matrices_json : str
        filepath of JSON of estimated times to travel between stops of routes.
    invalid_scores_json : str
        filepath of JSON of scores assigned to routes if they are invalid.
    **kwargs :
        Inputs placed in output. Intended for testing_time_seconds and
        training_time_seconds

    Returns
    -------
    scores : dict
        Dictionary containing submission score, individual route scores, feasibility
        of routes, and kwargs.

    '''

    actual_routes=read_json_data(actual_routes_json)
    good_format(actual_routes,'actual',actual_routes_json)
    submission=read_json_data(submission_json)
    good_format(submission,'proposed',submission_json)
    cost_matrices=read_json_data(cost_matrices_json)
    good_format(cost_matrices,'costs',cost_matrices_json)
    invalid_scores=read_json_data(invalid_scores_json)
    good_format(invalid_scores,'invalids',invalid_scores_json)

    df_new_travel = pd.DataFrame(cost_matrices)
    df_new_actual_routes = pd.DataFrame(actual_routes)
    df_new_proposed_routes = pd.DataFrame(submission)

    scores={'submission_score': 'x', 'route_scores': {}, 'route_SD': {}, 'route_ERP_norm': {}, 'route_ERP_ratio': {},
            'route_ERP_edit': {}, 'route_feasibility': {}, 'zone_route_scores': {}, 'route_quality': {},
            'total_travel_time_actual': {}, 'total_travel_time_proposed': {}, 'total_travel_time_ratio': {},
            'total_travel_time_diff': {}}

    for kwarg in kwargs:
        scores[kwarg]=kwargs[kwarg]

    for route in actual_routes:

        zone_actual_list, total_travel_time_actual = optimal_zone_sequence_scr(route, df_new_route, df_new_actual_routes, df_new_travel, 'actual')
        zone_proposed_list, total_travel_time_proposed = optimal_zone_sequence_scr(route, df_new_route, df_new_proposed_routes, df_new_travel, 'proposed')
        scores['zone_route_scores'][route] = seq_dev_zone(zone_actual_list,zone_proposed_list)
        scores['route_quality'][route] = df_new_route[route]['route_score']
        scores['total_travel_time_actual'][route] = total_travel_time_actual
        scores['total_travel_time_proposed'][route] = total_travel_time_proposed
        scores['total_travel_time_ratio'][route] = total_travel_time_actual / total_travel_time_proposed
        scores['total_travel_time_diff'][route] = total_travel_time_actual - total_travel_time_proposed

        if route not in submission:
            scores['route_scores'][route] = invalid_scores[route]
            scores['route_SD'][route] = invalid_scores[route]
            scores['route_ERP_norm'][route] = invalid_scores[route]
            scores['route_ERP_edit'][route] = invalid_scores[route]
            scores['route_ERP_ratio'][route] = invalid_scores[route]
            scores['route_feasibility'][route] = False
        else:
            actual_dict=actual_routes[route]
            actual=route2list(actual_dict)
            try:
                sub_dict=submission[route]
                sub=route2list(sub_dict)
            except:
                scores['route_scores'][route] = invalid_scores[route]
                scores['route_SD'][route] = invalid_scores[route]
                scores['route_ERP_norm'][route] = invalid_scores[route]
                scores['route_ERP_edit'][route] = invalid_scores[route]
                scores['route_ERP_ratio'][route] = invalid_scores[route]
                scores['route_feasibility'][route] = False
            else:
                if isinvalid(actual,sub):
                    scores['route_scores'][route] = invalid_scores[route]
                    scores['route_SD'][route] = invalid_scores[route]
                    scores['route_ERP_norm'][route] = invalid_scores[route]
                    scores['route_ERP_edit'][route] = invalid_scores[route]
                    scores['route_feasibility'][route] = False
                else:
                     cost_mat = cost_matrices[route]
                     route_score, route_SD, ERP_norm, ERP_edit = score(actual,sub,cost_mat)
                     scores['route_scores'][route] = route_score
                     scores['route_SD'][route] = route_SD
                     scores['route_ERP_norm'][route] = ERP_norm
                     scores['route_ERP_edit'][route] = ERP_edit
                     scores['route_ERP_ratio'][route] = ERP_norm / ERP_edit
                     scores['route_feasibility'][route] = True
    
    submission_score = np.mean(list(scores['route_scores'].values()))
    scores['submission_score'] = submission_score 

    scores['zone_SD_score'] = np.mean(list(scores['zone_route_scores'].values()))
    scores['SD_score'] = np.mean(list(scores['route_SD'].values()))
    scores['ERP_norm_score'] = np.mean(list(scores['route_ERP_norm'].values()))
    scores['ERP_edit_score'] = np.mean(list(scores['route_ERP_edit'].values()))
    scores['ERP_ratio_score'] = np.mean(list(scores['route_ERP_ratio'].values()))
    scores['travel_time_actual'] = np.mean(list(scores['total_travel_time_actual'].values()))
    scores['travel_time_proposed'] = np.mean(list(scores['total_travel_time_proposed'].values()))
    scores['travel_time_ratio'] = np.mean(list(scores['total_travel_time_ratio'].values()))
    scores['travel_time_diff'] = np.mean(list(scores['total_travel_time_diff'].values()))

    low_quality_routes = [route for route in scores['route_quality'] if scores['route_quality'][route] == 'Low']
    medium_quality_routes = [route for route in scores['route_quality'] if scores['route_quality'][route] == 'Medium']
    high_quality_routes = [route for route in scores['route_quality'] if scores['route_quality'][route] == 'High']

    if len(low_quality_routes):
        scores['low_quality_score'] = np.mean([scores['route_scores'][route] for route in low_quality_routes])
        scores['low_quality_SD'] = np.mean([scores['route_SD'][route] for route in low_quality_routes])
        scores['low_quality_ERP_norm'] = np.mean([scores['route_ERP_norm'][route] for route in low_quality_routes])
        scores['low_quality_ERP_edit'] = np.mean([scores['route_ERP_edit'][route] for route in low_quality_routes])
        scores['low_quality_ERP_ratio'] = np.mean([scores['route_ERP_ratio'][route] for route in low_quality_routes])
        scores['low_quality_zone_SD'] = np.mean([scores['zone_route_scores'][route] for route in low_quality_routes])
        scores['low_quality_tt_actual'] = np.mean([scores['total_travel_time_actual'][route] for route in low_quality_routes])
        scores['low_quality_tt_proposed'] = np.mean([scores['total_travel_time_proposed'][route] for route in low_quality_routes])
        scores['low_quality_tt_ratio'] = np.mean([scores['total_travel_time_ratio'][route] for route in low_quality_routes])
        scores['low_quality_tt_diff'] = np.mean([scores['total_travel_time_diff'][route] for route in low_quality_routes])
    else:
        scores['low_quality_score'] = 'NA'
        scores['low_quality_SD'] = 'NA'
        scores['low_quality_ERP_norm'] = 'NA'
        scores['low_quality_ERP_edit'] = 'NA'
        scores['low_quality_ERP_ratio'] = 'NA'
        scores['low_quality_zone_SD'] = 'NA'
        scores['low_quality_tt_actual'] = 'NA'
        scores['low_quality_tt_proposed'] = 'NA'
        scores['low_quality_tt_ratio'] = 'NA'
        scores['low_quality_tt_diff'] = 'NA'

    if len(medium_quality_routes):
        scores['medium_quality_score'] = np.mean([scores['route_scores'][route] for route in medium_quality_routes])
        scores['medium_quality_SD'] = np.mean([scores['route_SD'][route] for route in medium_quality_routes])
        scores['medium_quality_ERP_norm'] = np.mean([scores['route_ERP_norm'][route] for route in medium_quality_routes])
        scores['medium_quality_ERP_edit'] = np.mean([scores['route_ERP_edit'][route] for route in medium_quality_routes])
        scores['medium_quality_ERP_ratio'] = np.mean([scores['route_ERP_ratio'][route] for route in medium_quality_routes])
        scores['medium_quality_zone_SD'] = np.mean([scores['zone_route_scores'][route] for route in medium_quality_routes])
        scores['medium_quality_tt_actual'] = np.mean([scores['total_travel_time_actual'][route] for route in medium_quality_routes])
        scores['medium_quality_tt_proposed'] = np.mean([scores['total_travel_time_proposed'][route] for route in medium_quality_routes])
        scores['medium_quality_tt_ratio'] = np.mean([scores['total_travel_time_ratio'][route] for route in medium_quality_routes])
        scores['medium_quality_tt_diff'] = np.mean([scores['total_travel_time_diff'][route] for route in medium_quality_routes])
    else:
        scores['medium_quality_score'] = 'NA'
        scores['medium_quality_SD'] = 'NA'
        scores['medium_quality_ERP_norm'] = 'NA'
        scores['medium_quality_ERP_edit'] = 'NA'
        scores['medium_quality_ERP_ratio'] = 'NA'
        scores['medium_quality_zone_SD'] = 'NA'  
        scores['medium_quality_tt_actual'] = 'NA'
        scores['medium_quality_tt_proposed'] = 'NA'
        scores['medium_quality_tt_ratio'] = 'NA'
        scores['medium_quality_tt_diff'] = 'NA'
    
    if len(high_quality_routes):
        scores['high_quality_score'] = np.mean([scores['route_scores'][route] for route in high_quality_routes])
        scores['high_quality_SD'] = np.mean([scores['route_SD'][route] for route in high_quality_routes])  
        scores['high_quality_ERP_norm'] = np.mean([scores['route_ERP_norm'][route] for route in high_quality_routes])  
        scores['high_quality_ERP_edit'] = np.mean([scores['route_ERP_edit'][route] for route in high_quality_routes])
        scores['high_quality_ERP_ratio'] = np.mean([scores['route_ERP_ratio'][route] for route in high_quality_routes])
        scores['high_quality_zone_SD'] = np.mean([scores['zone_route_scores'][route] for route in high_quality_routes])
        scores['high_quality_tt_actual'] = np.mean([scores['total_travel_time_actual'][route] for route in high_quality_routes])
        scores['high_quality_tt_proposed'] = np.mean([scores['total_travel_time_proposed'][route] for route in high_quality_routes])
        scores['high_quality_tt_ratio'] = np.mean([scores['total_travel_time_ratio'][route] for route in high_quality_routes])
        scores['high_quality_tt_diff'] = np.mean([scores['total_travel_time_diff'][route] for route in high_quality_routes])
    else:
        scores['high_quality_score'] = 'NA'
        scores['high_quality_SD'] = 'NA'
        scores['high_quality_ERP_norm'] = 'NA'
        scores['high_quality_ERP_edit'] = 'NA'
        scores['high_quality_ERP_ratio'] = 'NA'
        scores['high_quality_zone_SD'] = 'NA'
        scores['high_quality_tt_actual'] = 'NA'
        scores['high_quality_tt_proposed'] = 'NA'
        scores['high_quality_tt_ratio'] = 'NA'
        scores['high_quality_tt_diff'] = 'NA'

    return scores

def score(actual,sub,cost_mat,g=1000):
    '''
    Scores individual routes.

    Parameters
    ----------
    actual : list
        Actual route.
    sub : list
        Submitted route.
    cost_mat : dict
        Cost matrix.
    g : int/float, optional
        ERP gap penalty. Irrelevant if large and len(actual)==len(sub). The
        default is 1000.

    Returns
    -------
    float
        Accuracy score from comparing sub to actual.

    '''
    norm_mat=normalize_matrix(cost_mat)

    ERP_norm, ERP_edit = erp_per_edit(actual,sub,norm_mat,g)
    return seq_dev(actual,sub)*(ERP_norm / ERP_edit), seq_dev(actual,sub), ERP_norm, ERP_edit

def erp_per_edit(actual,sub,matrix,g=1000):
    '''
    Outputs ERP of comparing sub to actual divided by the number of edits involved
    in the ERP. If there are 0 edits, returns 0 instead.

    Parameters
    ----------
    actual : list
        Actual route.
    sub : list
        Submitted route.
    matrix : dict
        Normalized cost matrix.
    g : int/float, optional
        ERP gap penalty. The default is 1000.

    Returns
    -------
    int/float
        ERP divided by number of ERP edits or 0 if there are 0 edits.

    '''
    total,count=erp_per_edit_helper(actual,sub,matrix,g)
    if count==0:
        return 0, 0
    else:
        return total, count

def erp_per_edit_helper(actual,sub,matrix,g=1000,memo=None):
    '''
    Calculates ERP and counts number of edits in the process.

    Parameters
    ----------
    actual : list
        Actual route.
    sub : list
        Submitted route.
    matrix : dict
        Normalized cost matrix.
    g : int/float, optional
        Gap penalty. The default is 1000.
    memo : dict, optional
        For memoization. The default is None.

    Returns
    -------
    d : float
        ERP from comparing sub to actual.
    count : int
        Number of edits in ERP.

    '''
    if memo==None:
        memo={}
    actual_tuple=tuple(actual)
    sub_tuple=tuple(sub)
    if (actual_tuple,sub_tuple) in memo:
        d,count=memo[(actual_tuple,sub_tuple)]
        return d,count
    if len(sub)==0:
        d=gap_sum(actual,g)
        count=len(actual)
    elif len(actual)==0:
        d=gap_sum(sub,g)
        count=len(sub)
    else:
        head_actual=actual[0]
        head_sub=sub[0]
        rest_actual=actual[1:]
        rest_sub=sub[1:]
        score1,count1=erp_per_edit_helper(rest_actual,rest_sub,matrix,g,memo)
        score2,count2=erp_per_edit_helper(rest_actual,sub,matrix,g,memo)
        score3,count3=erp_per_edit_helper(actual,rest_sub,matrix,g,memo)
        option_1=score1+dist_erp(head_actual,head_sub,matrix,g)
        option_2=score2+dist_erp(head_actual,'gap',matrix,g)
        option_3=score3+dist_erp(head_sub,'gap',matrix,g)
        d=min(option_1,option_2,option_3)
        if d==option_1:
            if head_actual==head_sub:
                count=count1
            else:
                count=count1+1
        elif d==option_2:
            count=count2+1
        else:
            count=count3+1
    memo[(actual_tuple,sub_tuple)]=(d,count)
    return d,count

def normalize_matrix(mat):
    '''
    Normalizes cost matrix.

    Parameters
    ----------
    mat : dict
        Cost matrix.

    Returns
    -------
    new_mat : dict
        Normalized cost matrix.

    '''
    new_mat=mat.copy()
    time_list=[]
    for origin in mat:
        for destination in mat[origin]:
            time_list.append(mat[origin][destination])
    avg_time=np.mean(time_list)
    std_time=np.std(time_list)
    min_new_time=np.inf
    for origin in mat:
        for destination in mat[origin]:
            old_time=mat[origin][destination]
            new_time=(old_time-avg_time)/std_time
            if new_time<min_new_time:
                min_new_time=new_time
            new_mat[origin][destination]=new_time
    for origin in new_mat:
        for destination in new_mat[origin]:
            new_time=new_mat[origin][destination]
            shifted_time=new_time-min_new_time
            new_mat[origin][destination]=shifted_time
    return new_mat

def gap_sum(path,g):
    '''
    Calculates ERP between two sequences when at least one is empty.

    Parameters
    ----------
    path : list
        Sequence that is being compared to an empty sequence.
    g : int/float
        Gap penalty.

    Returns
    -------
    res : int/float
        ERP between path and an empty sequence.

    '''
    res=0
    for p in path:
        res+=g
    return res

def dist_erp(p_1,p_2,mat,g=1000):
    '''
    Finds cost between two points. Outputs g if either point is a gap.

    Parameters
    ----------
    p_1 : str
        ID of point.
    p_2 : str
        ID of other point.
    mat : dict
        Normalized cost matrix.
    g : int/float, optional
        Gap penalty. The default is 1000.

    Returns
    -------
    dist : int/float
        Cost of substituting one point for the other.

    '''
    if p_1=='gap' or p_2=='gap':
        dist=g
    else:
        dist=mat[p_1][p_2]
    return dist

def seq_dev(actual,sub):
    '''
    Calculates sequence deviation.

    Parameters
    ----------
    actual : list
        Actual route.
    sub : list
        Submitted route.

    Returns
    -------
    float
        Sequence deviation.

    '''
    actual=actual[1:-1]
    sub=sub[1:-1]
    comp_list=[]
    for i in sub:
        comp_list.append(actual.index(i))
        comp_sum=0
    for ind in range(1,len(comp_list)):
        comp_sum+=abs(comp_list[ind]-comp_list[ind-1])-1
    n=len(actual)
    return (2/(n*(n-1)))*comp_sum

def isinvalid(actual,sub):
    '''
    Checks if submitted route is invalid.

    Parameters
    ----------
    actual : list
        Actual route.
    sub : list
        Submitted route.

    Returns
    -------
    bool
        True if route is invalid. False otherwise.

    '''
    if len(actual)!=len(sub) or set(actual)!=set(sub):
        return True
    elif actual[0]!=sub[0]:
        return True
    else:
        return False

def route2list(route_dict):
    '''
    Translates route from dictionary to list.

    Parameters
    ----------
    route_dict : dict
        Route as a dictionary.

    Returns
    -------
    route_list : list
        Route as a list.

    '''
    if 'proposed' in route_dict:
        stops=route_dict['proposed']
    elif 'actual' in route_dict:
        stops=route_dict['actual']
    route_list=[0]*(len(stops)+1)
    for stop in stops:
        route_list[stops[stop]]=stop
    route_list[-1]=route_list[0]
    return route_list
