import json
import os
import statistics

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# ----------------------------------------------------------------------------

# Структура данных
#
# Vec<Vec<EvolveOutputEachStep>>
# 
# struct EvolveOutputEachStep {
#     nn: Vec<f32>,
#     evals: Vec<TrackEvaluation>,
#     evals_cost: f32,
#     true_evals: Vec<TrackEvaluation>,
#     true_evals_cost: f32,
# }
# 
# struct TrackEvaluation {
#     name: String,
#     penalty: f32,
#     reward: f32,
#     early_finish_percent: f32,
#     distance_percent: f32,
#     rewards_acquired_percent: f32,
#     all_acquired: bool,
# }

# ----------------------------------------------------------------------------

data = read_json_file("nn2_output_discrete_action.json")

# data = sorted(data, key=lambda x: x[80]["evals_cost"])
for i, run in enumerate(data):
    if i % 100 == 0 or i == 999 or i == 29:
        last = run[-1]
        print("---------------------------------", i)
        print("NN:", json.dumps(last["nn"]))
        print("eval: ", last["evals_cost"])
