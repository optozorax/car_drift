import json
import os
import matplotlib.pyplot as plt
import numpy as np
import statistics

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

json_files = [f for f in os.listdir('.') if f.endswith('.json')]

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

all_tracks = [
    "straight",
    "straight_45",
    "turn_right_smooth",
    "smooth_left_and_right",
    "turn_left_90",
    "turn_left_180",
    "complex",

    # "straight_mirror",
    # "straight_45_mirror",
    # "turn_right_smooth_mirror",
    # "smooth_left_and_right_mirror",
    # "turn_left_90_mirror",
    # "turn_left_180_mirror",
    # "complex_mirror",
    # надо ли добавить mirror?
]

colors = [
    'tab:red',
    'tab:green',
    'tab:blue',
    'tab:orange',
    'tab:purple',
    'tab:brown',
    'tab:pink',
    'tab:gray',
    'tab:olive',
    'tab:cyan',
]

styles = [
    'solid',
    'dashed',
    'dotted',
    'dashdot',
    ':',
    '-.',
    '--',
]

# ----------------------------------------------------------------------------

info_field = "true_evals"

def step_get_eval_value(step):
    return step["evals_cost"]

def step_get_simple_physics(step):
    return step[info_field][0].get("simple_physics", 1.0)

def step_get_autoencoder_loss(step):
    return step[info_field][0].get("autoencoder_loss", 0.0)

def step_get_true_eval_value(step):
    return step["true_evals_cost"]

def step_get_track_completed(step, track):
    if [x for x in step[info_field] if x["name"] == track][0]["all_acquired"]:
        return 1.0
    else:
        return 0.0

def step_get_penalty_avg(step):
    return statistics.fmean([x["penalty"] for x in step[info_field]])

def step_get_penalty_max(step):
    return max([x["penalty"] for x in step[info_field]])

def step_get_early_finish_avg(step):
    return statistics.fmean([x["early_finish_percent"] for x in step[info_field]])

def step_get_distance_min(step):
    return min([x["distance_percent"] for x in step[info_field]])

def step_get_distance_avg(step):
    return statistics.fmean([x["distance_percent"] for x in step[info_field]])

# ----------------------------------------------------------------------------

def process_data(data, function):
    return [[function(y) for y in x] for x in data]

def data_has_track(data, track):
    for run in data:
        for epoch in run:
            for tracks in epoch["true_evals"]:
                if tracks["name"] == track:
                    return True
            return False
    return False

def pad_losses_nan(losses):
    max_length = max(len(arr) for arr in losses)

    padded_losses = np.array([
        np.pad(arr, 
               (0, max_length - len(arr)), 
               mode='constant', 
               constant_values=np.nan) 
        for arr in losses
    ])
    
    return padded_losses

def pad_losses_repeat_last(losses):
    max_length = max(len(arr) for arr in losses)
    
    padded_losses = np.array([
        np.pad(arr, 
               (0, max_length - len(arr)), 
               mode='edge')
        for arr in losses
    ])
    
    return padded_losses

def draw_all_values(axs, losses, color, alpha_mul=1.0):
    for loss_arr in losses:
        axs.plot(loss_arr, alpha=0.3 * alpha_mul, color=color)

def draw_sum_values(axs, losses, name, color=None, alpha_mul=1.0, linestyle='solid'):
    # losses = pad_losses_nan(losses)
    losses = pad_losses_repeat_last(losses)
    losses_sum = np.average(losses, axis=0)
    axs.plot(losses_sum, alpha=alpha_mul, label=name, color=color, linestyle=linestyle)

def draw_values_mean_std(axs, losses, name, color, alpha_mul=1.0, disable_percentiles=False):
    # losses = pad_losses_nan(losses)
    losses = pad_losses_repeat_last(losses)
    losses_median = np.nanmedian(losses, axis=0)
    losses_10 = np.nanpercentile(losses, 10, axis=0)
    losses_25 = np.nanpercentile(losses, 25, axis=0)
    losses_75 = np.nanpercentile(losses, 75, axis=0)
    losses_90 = np.nanpercentile(losses, 90, axis=0)

    x = range(len(losses_median))
    axs.plot(losses_median, label=name, color=color, alpha=alpha_mul)
    axs.fill_between(x, losses_25, losses_75, color=color, alpha=0.2 * alpha_mul)
    if not disable_percentiles:
        axs.fill_between(x, losses_10, losses_25, color=color, alpha=0.05 * alpha_mul)
        axs.fill_between(x, losses_75, losses_90, color=color, alpha=0.05 * alpha_mul)

def draw_datas(axs, datas_with_style, skip_individuals, only_complex_track, disable_percentiles):
    offset = 0
    if not skip_individuals:
        offset = 1
        draw_all_values(axs[0, 0], process_data(datas_with_style[0]["data"], step_get_eval_value), 'blue')
        axs[0, 0].set_title('Training Loss - Individual Runs')
        axs[0, 0].set(xlabel='Epoch', ylabel='Loss')

        draw_all_values(axs[0, 1], process_data(datas_with_style[0]["data"], step_get_true_eval_value), 'red')
        axs[0, 1].set_title('Validation Loss - Individual Runs')
        axs[0, 1].set(xlabel='Epoch', ylabel='Loss')

    if True:
    # if False:
        for data in datas_with_style:
            draw_values_mean_std(axs[offset, 0], process_data(data["data"], step_get_eval_value), data["name"], data.get('color', 'blue'), alpha_mul=data.get('alpha', 1.0), disable_percentiles=disable_percentiles)
        axs[offset, 0].set_title('Training Loss - Mean + Std')
        axs[offset, 0].set(xlabel='Epoch', ylabel='Loss')
        axs[offset, 0].legend()

        for data in datas_with_style:
            draw_values_mean_std(axs[offset, 1], process_data(data["data"], step_get_true_eval_value), data["name"], data.get('color', 'red'), alpha_mul=data.get('alpha', 1.0), disable_percentiles=disable_percentiles)
        axs[offset, 1].set_title('Validating Loss - Mean + Std')
        axs[offset, 1].set(xlabel='Epoch', ylabel='Loss')
        axs[offset, 1].legend()
    else:
        offset = -1

    for (i, track) in enumerate(all_tracks):
        color = colors[i]
        for i, data in enumerate(datas_with_style):
            name = track
            if i != 0:
                name = None
            if data_has_track(data["data"], track):
                if only_complex_track:
                    if track == "complex":
                        draw_sum_values(axs[offset+1, 0], process_data(data["data"], lambda x: step_get_track_completed(x, track)), data["name"], color=data["color"], alpha_mul=data.get('alpha', 1.0), linestyle=data.get('style', None))
                else:
                    draw_sum_values(axs[offset+1, 0], process_data(data["data"], lambda x: step_get_track_completed(x, track)), name, color=color, alpha_mul=data.get('alpha', 1.0), linestyle=data.get('style', None))
    axs[offset+1, 0].set_title('Track completion stats')
    axs[offset+1, 0].set(xlabel='Epoch', ylabel='How many evolutions completed track')
    axs[offset+1, 0].legend()

    for data in datas_with_style:
        draw_values_mean_std(axs[offset+1, 1], process_data(data["data"], step_get_early_finish_avg), data["name"], data.get('color', 'green'), alpha_mul=data.get('alpha', 1.0), disable_percentiles=disable_percentiles)
    axs[offset+1, 1].set_title('Early finish average')
    axs[offset+1, 1].set(xlabel='Epoch', ylabel='Percent')
    axs[offset+1, 1].legend()

    for data in datas_with_style:
        draw_values_mean_std(axs[offset+2, 0], process_data(data["data"], step_get_penalty_avg), data["name"], data.get('color', 'green'), alpha_mul=data.get('alpha', 1.0), disable_percentiles=disable_percentiles)
    axs[offset+2, 0].set_title('Penalty average')
    axs[offset+2, 0].set(xlabel='Epoch', ylabel='Penalty')
    axs[offset+2, 0].legend()
    axs[offset+2, 0].set_ylim(top=25)

    for data in datas_with_style:
        draw_values_mean_std(axs[offset+2, 1], process_data(data["data"], step_get_penalty_max), data["name"], data.get('color', 'red'), alpha_mul=data.get('alpha', 1.0), disable_percentiles=disable_percentiles)
    axs[offset+2, 1].set_title('Penalty max')
    axs[offset+2, 1].set(xlabel='Epoch', ylabel='Penalty')
    axs[offset+2, 1].legend()
    # axs[offset+2, 1].set_ylim(top=25)

    for data in datas_with_style:
        draw_values_mean_std(axs[offset+3, 0], process_data(data["data"], step_get_distance_avg), data["name"], data.get('color', 'green'), alpha_mul=data.get('alpha', 1.0), disable_percentiles=disable_percentiles)
    axs[offset+3, 0].set_title('Distance average')
    axs[offset+3, 0].set(xlabel='Epoch', ylabel='Percent')
    axs[offset+3, 0].legend()

    for data in datas_with_style:
        draw_values_mean_std(axs[offset+3, 1], process_data(data["data"], step_get_distance_min), data["name"], data.get('color', 'blue'), alpha_mul=data.get('alpha', 1.0), disable_percentiles=disable_percentiles)
    axs[offset+3, 1].set_title('Distance min')
    axs[offset+3, 1].set(xlabel='Epoch', ylabel='Percent')
    axs[offset+3, 1].legend()

    for data in datas_with_style:
        draw_values_mean_std(axs[offset+4, 0], process_data(data["data"], step_get_simple_physics), data["name"], data.get('color', 'blue'), alpha_mul=data.get('alpha', 1.0), disable_percentiles=disable_percentiles)
    axs[offset+4, 0].set_title('Simple physics value')
    axs[offset+4, 0].set(xlabel='Epoch', ylabel='Percent')
    axs[offset+4, 0].legend()

    for data in datas_with_style:
        draw_values_mean_std(axs[offset+4, 1], process_data(data["data"], step_get_autoencoder_loss), data["name"], data.get('color', 'blue'), alpha_mul=data.get('alpha', 1.0), disable_percentiles=disable_percentiles)
    axs[offset+4, 1].set_title('Autoencoder loss average between tracks')
    axs[offset+4, 1].set(xlabel='Epoch', ylabel='Loss')
    axs[offset+4, 1].legend()

def draw_graph(datas, title, filename, skip_individuals=False, only_complex_track=False, disable_percentiles=False):
    lines = 5
    if skip_individuals:
        fig, axs = plt.subplots(lines, 2, figsize=(15, 5 * lines))
    else:
        fig, axs = plt.subplots(lines+1, 2, figsize=(15, 5 * (lines+1)))
    fig.suptitle(title)
    draw_datas(axs, datas, skip_individuals, only_complex_track, disable_percentiles)
    for ax in axs.flat:
        ax.set_xlim(left=-30, right=530)
        ax.grid(axis='both', color='0.85')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# ----------------------------------------------------------------------------

# draw_graph(
#     [
#         {
#             "data": read_json_file("./default.json"),
#             "name": "cma-es",
#             "alpha": 0.5,
#             "color": colors[0],
#             "style": styles[0],
#         },
#         {
#             "data": read_json_file("./differential_evolution.json"),
#             "name": "differential evolution",
#             "alpha": 0.5,
#             "color": colors[1],
#             'style': styles[1],
#         },
#         {
#             "data": read_json_file("./particle_swarm.json"),
#             "name": "particle swarm",
#             "alpha": 0.5,
#             "color": colors[2],
#             'style': styles[2],
#         },
#     ],
#     "Optimization algorithms",
#     "_optimization_algorithms.png",
#     skip_individuals=True,
# )

# draw_graph(
#     [
#         {
#             "data": read_json_file("./second_way.json"),
#             "name": "cma-es",
#             "alpha": 0.5,
#             "color": colors[0],
#             "style": styles[0],
#         },
#         {
#             "data": read_json_file("./differential_evolution_2w.json"),
#             "name": "differential evolution",
#             "alpha": 0.5,
#             "color": colors[1],
#             'style': styles[1],
#         },
#         {
#             "data": read_json_file("./particle_swarm_2w.json"),
#             "name": "particle swarm",
#             "alpha": 0.5,
#             "color": colors[2],
#             'style': styles[2],
#         },
#     ],
#     "Optimization algorithms, second way metric",
#     "_optimization_algorithms_2w.png",
#     skip_individuals=True,
# )

# draw_graph(
#     [
#         {
#             "data": read_json_file("./penalty_0.json"),
#             "name": "0 penalty",
#             "color": colors[0],
#         },
#         {
#             "data": read_json_file("./penalty_10.json"),
#             "name": "10 penalty",
#             "color": colors[1],
#         },
#         {
#             "data": read_json_file("./penalty_50.json"),
#             "name": "50 penalty",
#             "color": colors[2],
#         },
#         {
#             "data": read_json_file("./penalty_100.json"),
#             "name": "100 penalty",
#             "color": colors[3],
#         },
#         {
#             "data": read_json_file("./penalty_200.json"),
#             "name": "200 penalty",
#             "color": colors[4],
#         },
#         {
#             "data": read_json_file("./penalty_500.json"),
#             "name": "500 penalty",
#             "color": colors[5],
#         },
#         {
#             "data": read_json_file("./penalty_1000.json"),
#             "name": "1000 penalty",
#             "color": colors[6],
#         },
#     ],
#     "Penalties",
#     "_penalties.png",
#     skip_individuals=True,
#     only_complex_track=True,
#     disable_percentiles=True,
# )

# draw_graph(
#     [
#         {
#             "data": read_json_file("./random_output_no.json"),
#             "name": "0 random output",
#             "color": colors[0],
#         },
#         {
#             "data": read_json_file("./random_output_0.01.json"),
#             "name": "0.01 random output",
#             "color": colors[1],
#         },
#         {
#             "data": read_json_file("./random_output_0.05.json"),
#             "name": "0.05 random output",
#             "color": colors[2],
#         },
#         {
#             "data": read_json_file("./random_output_0.1.json"),
#             "name": "0.1 random output",
#             "color": colors[3],
#         },
#         {
#             "data": read_json_file("./random_output_0.2.json"),
#             "name": "0.2 random output",
#             "color": colors[4],
#         },
#         {
#             "data": read_json_file("./random_output_0.4.json"),
#             "name": "0.4 random output",
#             "color": colors[5],
#         },
#     ],
#     "Random output",
#     "_random_output.png",
#     skip_individuals=True,
#     only_complex_track=True,
#     disable_percentiles=True,
# )

# draw_graph(
#     [
#         {
#             "data": read_json_file("./random_output_no_2w.json"),
#             "name": "0 random output",
#             "color": colors[0],
#         },
#         {
#             "data": read_json_file("./random_output_0.01_2w.json"),
#             "name": "0.01 random output",
#             "color": colors[1],
#         },
#         {
#             "data": read_json_file("./random_output_0.05_2w.json"),
#             "name": "0.05 random output",
#             "color": colors[2],
#         },
#         {
#             "data": read_json_file("./random_output_0.1_2w.json"),
#             "name": "0.1 random output",
#             "color": colors[3],
#         },
#         {
#             "data": read_json_file("./random_output_0.2_2w.json"),
#             "name": "0.2 random output",
#             "color": colors[4],
#         },
#         {
#             "data": read_json_file("./random_output_0.4_2w.json"),
#             "name": "0.4 random output",
#             "color": colors[5],
#         },
#     ],
#     "Random output, second way metric",
#     "_random_output_2w.png",
#     skip_individuals=True,
#     only_complex_track=True,
#     disable_percentiles=True,
# )

# draw_graph(
#     [
#         {
#             "data": read_json_file("./reward_0.json"),
#             "name": "0 reward",
#             "color": colors[0],
#         },
#         {
#             "data": read_json_file("./reward_10.json"),
#             "name": "10 reward",
#             "color": colors[1],
#         },
#         {
#             "data": read_json_file("./reward_50.json"),
#             "name": "50 reward",
#             "color": colors[2],
#         },
#         {
#             "data": read_json_file("./reward_100.json"),
#             "name": "100 reward",
#             "color": colors[3],
#         },
#         {
#             "data": read_json_file("./reward_200.json"),
#             "name": "200 reward",
#             "color": colors[4],
#         },
#         {
#             "data": read_json_file("./reward_500.json"),
#             "name": "500 reward",
#             "color": colors[5],
#         },
#         {
#             "data": read_json_file("./reward_1000.json"),
#             "name": "1000 reward",
#             "color": colors[6],
#         },
#     ],
#     "Rewards",
#     "_rewards.png",
#     skip_individuals=True,
#     only_complex_track=True,
#     disable_percentiles=True,
# )

# draw_graph(
#     [
#         {
#             "data": read_json_file("./simple_physics_0.0.json"),
#             "name": "0 simple physics",
#             "alpha": 0.5,
#             "color": colors[0],
#             "style": styles[0],
#         },
#         {
#             "data": read_json_file("./simple_physics_0.5.json"),
#             "name": "0.5 simple physics",
#             "alpha": 0.5,
#             "color": colors[1],
#             'style': styles[1],
#         },
#         {
#             "data": read_json_file("./simple_physics_1.0.json"),
#             "name": "1.0 simple physics",
#             "alpha": 0.5,
#             "color": colors[2],
#             'style': styles[2],
#         },
#     ],
#     "Simple physics",
#     "_simple_physics.png",
#     skip_individuals=True,
# )

# draw_graph(
#     [
#         {
#             "data": read_json_file("./simple_physics_0.0_2w.json"),
#             "name": "0 simple physics",
#             "alpha": 0.5,
#             "color": colors[0],
#             "style": styles[0],
#         },
#         {
#             "data": read_json_file("./simple_physics_0.5_2w.json"),
#             "name": "0.5 simple physics",
#             "alpha": 0.5,
#             "color": colors[1],
#             'style': styles[1],
#         },
#         {
#             "data": read_json_file("./simple_physics_1.0_2w.json"),
#             "name": "1.0 simple physics",
#             "alpha": 0.5,
#             "color": colors[2],
#             'style': styles[2],
#         },
#     ],
#     "Simple physics, second way metric",
#     "_simple_physics_2w.png",
#     skip_individuals=True,
# )

# draw_graph(
#     [
#         {
#             "data": read_json_file("./stop_penalty_0.json"),
#             "name": "0 stop penalty",
#             "alpha": 0.5,
#             "color": colors[0],
#         },
#         {
#             "data": read_json_file("./stop_penalty_1.json"),
#             "name": "1 stop penalty",
#             "alpha": 0.5,
#             "color": colors[1],
#         },
#         {
#             "data": read_json_file("./stop_penalty_5.json"),
#             "name": "5 stop penalty",
#             "alpha": 0.5,
#             "color": colors[2],
#         },
#         {
#             "data": read_json_file("./stop_penalty_10.json"),
#             "name": "10 stop penalty",
#             "alpha": 0.5,
#             "color": colors[3],
#         },
#         {
#             "data": read_json_file("./stop_penalty_20.json"),
#             "name": "20 stop penalty",
#             "alpha": 0.5,
#             "color": colors[4],
#         },
#         {
#             "data": read_json_file("./stop_penalty_50.json"),
#             "name": "50 stop penalty",
#             "alpha": 0.5,
#             "color": colors[5],
#         },
#         {
#             "data": read_json_file("./stop_penalty_100.json"),
#             "name": "100 stop penalty",
#             "alpha": 0.5,
#             "color": colors[6],
#         },
#     ],
#     "Stop penalty",
#     "_stop_penalties.png",
#     skip_individuals=True,
#     only_complex_track=True,
#     disable_percentiles=True,
# )

# draw_graph(
#     [
#         {
#             "data": read_json_file("./stop_penalty_0_2w.json"),
#             "name": "0 stop penalty",
#             "alpha": 0.5,
#             "color": colors[0],
#         },
#         {
#             "data": read_json_file("./stop_penalty_1_2w.json"),
#             "name": "1 stop penalty",
#             "alpha": 0.5,
#             "color": colors[1],
#         },
#         {
#             "data": read_json_file("./stop_penalty_5_2w.json"),
#             "name": "5 stop penalty",
#             "alpha": 0.5,
#             "color": colors[2],
#         },
#         {
#             "data": read_json_file("./stop_penalty_10_2w.json"),
#             "name": "10 stop penalty",
#             "alpha": 0.5,
#             "color": colors[3],
#         },
#         {
#             "data": read_json_file("./stop_penalty_20_2w.json"),
#             "name": "20 stop penalty",
#             "alpha": 0.5,
#             "color": colors[4],
#         },
#         {
#             "data": read_json_file("./stop_penalty_50_2w.json"),
#             "name": "50 stop penalty",
#             "alpha": 0.5,
#             "color": colors[5],
#         },
#         {
#             "data": read_json_file("./stop_penalty_100_2w.json"),
#             "name": "100 stop penalty",
#             "alpha": 0.5,
#             "color": colors[6],
#         },
#     ],
#     "Stop penalty, second way metric",
#     "_stop_penalties_2w.png",
#     skip_individuals=True,
#     only_complex_track=True,
#     disable_percentiles=True,
# )

# draw_graph(
#     [
#         {
#             "data": read_json_file("./evolve_simple_2w.json")[:500],
#             "name": "adaptive simple physics",
#         }
#     ],
#     "evolve_simple_2w",
#     "evolve_simple_2w.png",
# )

# draw_graph(
#     [
#         {
#             "data": read_json_file("./evolve_simple_2w_next.json"),
#             "name": "adaptive simple physics 2",
#         }
#     ],
#     "evolve_simple_2w_next",
#     "evolve_simple_2w_next.png",
# )

# draw_graph(
#     [
#         {
#             "data": read_json_file("./evolve_simple_2w_next.json"),
#             "name": "adaptive simple physics 2",
#             "color": 'gray',
#             "alpha": 0.5,
#             'style': 'dashed',
#         },
#         {
#             "data": read_json_file("./evolve_simple_2w_next_internals.json"),
#             "name": "with internals",
#         }
#     ],
#     "evolve_simple_2w_next_internals",
#     "evolve_simple_2w_next_internals.png",
# )

# draw_graph(
#     [
#         {
#             "data": read_json_file("./evolve_simple_2w_next.json"),
#             "name": "adaptive simple physics 2",
#             "color": 'gray',
#             "alpha": 0.5,
#             'style': 'dashed',
#         },
#         {
#             "data": read_json_file("./evolve_simple_2w_next_value.json"),
#             "name": "with simple physics value",
#         }
#     ],
#     "evolve_simple_2w_next_value",
#     "evolve_simple_2w_next_value.png",
# )

# draw_graph(
#     [
#         {
#             "data": read_json_file("./evolve_simple_2w_next.json"),
#             "name": "adaptive simple physics 2",
#             "color": 'gray',
#             "alpha": 0.5,
#             'style': 'dashed',
#         },
#         {
#             "data": read_json_file("./evolve_simple_2w_next_both.json"),
#             "name": "with both",
#         }
#     ],
#     "evolve_simple_2w_next_both",
#     "evolve_simple_2w_next_both.png",
# )

# draw_graph(
#     [
#         {
#             "data": read_json_file("./evolve_simple_NEW.json"),
#             "name": "for loop to evolve simple",
#             "alpha": 0.5,
#             "color": colors[0],
#         },
#         {
#             "data": read_json_file("./evolve_simple_2w_NEW.json"),
#             "name": "insert simplicity into optimization",
#             "alpha": 0.5,
#             "color": colors[1],
#         },
#         {
#             "data": read_json_file("./simple_physics_0.0_2w_NEW.json"),
#             "name": "start from hard physics",
#             "alpha": 0.5,
#             "color": colors[2],
#         },
#     ],
#     "Different approaches to evolve to 0.0 simple value",
#     "_evolve_simple.png",
#     skip_individuals=True,
#     only_complex_track=True,
#     disable_percentiles=True,
# )

# draw_graph(
#     [
#         {
#             "data": read_json_file("./evolve_simple_NEW.json"),
#             "name": "for loop to evolve simple",
#         },
#     ],
#     "evolve_simple_NEW",
#     "evolve_simple_NEW_2.png",
#     skip_individuals=True,
# )


# draw_graph(
#     [
#         {
#             "data": read_json_file("./simple_physics_0.0_2w_evolve.json")[:500],
#             "name": "adaptive simple physics",
#         }
#     ],
#     "simple_physics_0.0_2w_evolve",
#     "simple_physics_0.0_2w_evolve.png",
# )

# draw_graph(
#     [
#         {
#             "data": read_json_file("./evolve_simple_2w.json")[:500],
#             "name": "adaptive simple physics",
#             "alpha": 0.5,
#             "color": colors[0],
#             "style": styles[0],
#         },
#         {
#             "data": read_json_file("./simple_physics_0.0_2w_evolve.json")[:500],
#             "name": "simple physics 0.0",
#             "alpha": 0.5,
#             "color": colors[2],
#             'style': styles[1],
#         },
#     ],
#     "Adaptive vs static simple physics",
#     "_adaptive_simple_physics.png",
#     skip_individuals=True,
# )

# data_default = read_json_file("./default.json")
# data_2w = read_json_file("./second_way.json")
data_nn = read_json_file("./nn_default.json")
for file_name in json_files:
    # if file_name != "evolve_simple_2w.json":
    #     continue
    file_path = os.path.join('.', file_name)
    data = read_json_file(file_path)
    if file_name.startswith("nn_"):
        draw_graph(
            [
                {
                    "data": data,
                    "name": "current",
                },
                {
                    "data": data_nn,
                    "name": "default",
                    "color": 'gray',
                    "alpha": 0.5,
                    'style': 'dashed',
                }
            ],
            f'Graphs for {file_name}',
            f'{file_name}_graphs.png',
        )
    # else:
        # draw_graph(
        #     [
        #         {
        #             "data": data,
        #             "name": "current",
        #         },
        #         {
        #             "data": data_default,
        #             "name": "default",
        #             "color": 'gray',
        #             "alpha": 0.5,
        #             'style': 'dashed',
        #         }
        #     ],
        #     f'Graphs for {file_name}',
        #     f'{file_name}_graphs.png',
        # )
    print(f"Finish drawing for {file_name}")
