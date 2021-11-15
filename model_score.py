import os, json, sys, time
# Import local score file
import score
import pandas as pd
from os import path

# Read JSON data from the given filepath
def read_json_data(filepath):
    try:
        with open(filepath, newline = '') as in_file:
            return json.load(in_file)
    except FileNotFoundError:
        print("The '{}' file is missing!".format(filepath))
    except json.JSONDecodeError:
        print("Error in the '{}' JSON data!".format(filepath))
    except Exception as e:
        print("Error when reading the '{}' file!".format(filepath))
        print(e)
    return None

if __name__ == '__main__':

    start = time.time()
    data_folder = 'data'
    if len(sys.argv) > 1:
        data_folder = sys.argv[1]
    if len(sys.argv) == 3:
        omega = float(sys.argv[2])
        omega_start = omega
        omega_end = omega
        print(f'omega={omega}')
    elif len(sys.argv) == 5:
        omega = float(sys.argv[2])
        omega_start = float(sys.argv[3])
        omega_end = float(sys.argv[4])
        print(f'omegas={omega_start,omega,omega_end}')

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Read JSON time inputs
    model_build_time = read_json_data(os.path.join(BASE_DIR, f'{data_folder}/model_score_timings/model_build_time.json'))
    model_apply_time = read_json_data(os.path.join(BASE_DIR, f'{data_folder}/model_score_timings/model_apply_time.json'))
    df_new_route = pd.read_json(path.join(BASE_DIR, f'{data_folder}/model_apply_inputs/new_route_data.json'))

    submission_json = os.path.join(BASE_DIR, f'{data_folder}/model_apply_outputs/proposed_sequences.json')
    # if len(sys.argv) == 3:
        # submission_json = os.path.join(BASE_DIR, f'{data_folder}/model_apply_outputs/proposed_sequences_omega_{omega:.2f}.json')
    # elif len(sys.argv) == 5:
    submission_json = os.path.join(BASE_DIR, f'{data_folder}/model_apply_outputs/proposed_sequences_omegas_{omega_start:.2f}-{omega:.2f}-{omega_end:.2f}.json')
        

    print('Beginning Score Evaluation... ', end='')
    output = score.evaluate(
        df_new_route,
        actual_routes_json = os.path.join(BASE_DIR, f'{data_folder}/model_score_inputs/new_actual_sequences.json'),
        invalid_scores_json = os.path.join(BASE_DIR, f'{data_folder}/model_score_inputs/new_invalid_sequence_scores.json'),
        submission_json=submission_json,
        cost_matrices_json = os.path.join(BASE_DIR, f'{data_folder}/model_apply_inputs/new_travel_times.json'),
        # model_apply_time = model_apply_time.get("time"),
        # model_build_time = model_build_time.get("time")
    )

    # Write Outputs to File
    output_path = os.path.join(BASE_DIR, f'{data_folder}/model_score_outputs/scores.json')
    if len(sys.argv) == 3:
        output_path = os.path.join(BASE_DIR, f'{data_folder}/model_score_outputs/scores_omega_{omega:.2f}.json')
    elif len(sys.argv) == 5:
        output_path = os.path.join(BASE_DIR, f'{data_folder}/model_score_outputs/scores_omegas_{omega_start:.2f}-{omega:.2f}-{omega_end:.2f}.json')
    
    with open(output_path, 'w') as out_file:
        json.dump(output, out_file)

    # Print Pretty Output
    rt_show=output.get('route_scores')
    extra_str=None
    if len(rt_show.keys())>5:
        rt_show=dict(list(rt_show.items())[:5])
        extra_str="..."
        print("\nFirst five route_scores:")
    else:
        print("\nAll route_scores:")
    for rt_key, rt_score in rt_show.items():
        print(rt_key,": ",rt_score)
    if extra_str:
        print(extra_str)

    print('Route Scores all quality:')
    print(f"Route score: {output.get('submission_score')}")
    print(f"SD: {output.get('SD_score')}")
    print(f"ERP_norm: {output.get('ERP_norm_score')}")
    print(f"ERP_edit: {output.get('ERP_edit_score')}")
    print(f"ERP_ratio: {output.get('ERP_ratio_score')}")
    print(f"Zone SD: {output.get('zone_SD_score')}\n")
    print(f"TT Actual: {output.get('travel_time_actual')}")
    print(f"TT Proposed: {output.get('travel_time_proposed')}")
    print(f"TT Ratio: {output.get('travel_time_ratio')}")
    print(f"TT Diff: {output.get('travel_time_diff')}")
    
    print(f'\n\n Low Quality Routes:')
    print(f"Route score: {output.get('low_quality_score')}")
    print(f"SD: {output.get('low_quality_SD')}")
    print(f"ERP_norm: {output.get('low_quality_ERP_norm')}")
    print(f"ERP_edit: {output.get('low_quality_ERP_edit')}")
    print(f"ERP_ratio: {output.get('low_quality_ERP_ratio')}")
    print(f"Zone SD: {output.get('low_quality_zone_SD')}\n")
    print(f"TT Actual: {output.get('low_quality_tt_actual')}")
    print(f"TT Proposed: {output.get('low_quality_tt_proposed')}")
    print(f"TT Ratio: {output.get('low_quality_tt_ratio')}")
    print(f"TT Diff: {output.get('low_quality_tt_diff')}")

    print(f'\n\n Medium Quality Routes:')
    print(f"Route score: {output.get('medium_quality_score')}")
    print(f"SD: {output.get('medium_quality_SD')}")
    print(f"ERP_norm: {output.get('medium_quality_ERP_norm')}")
    print(f"ERP_edit: {output.get('medium_quality_ERP_edit')}")
    print(f"ERP_ratio: {output.get('medium_quality_ERP_ratio')}")
    print(f"Zone SD: {output.get('medium_quality_zone_SD')}\n")
    print(f"TT Actual: {output.get('medium_quality_tt_actual')}")
    print(f"TT Proposed: {output.get('medium_quality_tt_proposed')}")
    print(f"TT Ratio: {output.get('medium_quality_tt_ratio')}")
    print(f"TT Diff: {output.get('medium_quality_tt_diff')}")

    print(f'\n\n High Quality Routes:')
    print(f"Route score: {output.get('high_quality_score')}")
    print(f"SD: {output.get('high_quality_SD')}")
    print(f"ERP_norm: {output.get('high_quality_ERP_norm')}")
    print(f"ERP_edit: {output.get('high_quality_ERP_edit')}")
    print(f"ERP_ratio: {output.get('high_quality_ERP_ratio')}")
    print(f"Zone SD: {output.get('high_quality_zone_SD')}\n")
    print(f"TT Actual: {output.get('high_quality_tt_actual')}")
    print(f"TT Proposed: {output.get('high_quality_tt_proposed')}")
    print(f"TT Ratio: {output.get('high_quality_tt_ratio')}")
    print(f"TT Diff: {output.get('high_quality_tt_diff')}")

    print(f'Duration: {time.time() - start}')
    
