import argparse
import importlib
import json
import os
import pickle
import random

os.environ["MUJOCO_GL"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"

import numpy as np
import jax
import jax.numpy as jnp
import flax.serialization

from agents.fbpiswitch import FBpiSwitchAgent, get_config
from utils.datasets import Dataset
from utils.env_utils import make_env_and_datasets, relabel_dataset
from utils.evaluation import evaluate
from graph_planner import FBGraphBuilder, GraphPlanner, GraphPlannerAgent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--env_name", type=str,
                        default="ogbench-antmaze-medium-navigate-v0")
    parser.add_argument("--num_episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    # Параметры графа
    parser.add_argument("--k_neighbors", type=int, default=10)
    parser.add_argument("--graph_metric", type=str, default="cosine")
    parser.add_argument("--subsample_stride", type=int, default=5)
    # Параметры планировщика
    parser.add_argument("--switching_threshold", type=float, default=0.85)
    parser.add_argument("--max_subgoals", type=int, default=8)
    
    return parser.parse_args()


def load_config(checkpoint_dir):
    config = get_config()

    flags_path = os.path.join(checkpoint_dir, "flags.json")
    with open(flags_path, "r") as f:
        saved_flags = json.load(f)

    agent_cfg = saved_flags.get("agent", {})
    for key, value in agent_cfg.items():
        if value is not None:
            config[key] = value

    if config.get("frame_stack") is None:
        config["frame_stack"] = 1

    return config



def load_checkpoint(agent, checkpoint_dir):
    params_path = os.path.join(checkpoint_dir, "params.pkl")
    with open(params_path, "rb") as f:
        load_dict = pickle.load(f)

    agent_state = load_dict.get("agent", load_dict)

    agent = flax.serialization.from_state_dict(agent, agent_state)

    return agent


def main():
    args = parse_args()

    config = load_config(args.checkpoint_dir)
    random.seed(args.seed)
    np.random.seed(args.seed)

    eval_env, train_dataset_dict, val_dataset_dict = make_env_and_datasets(
        args.env_name, frame_stack=config["frame_stack"], add_info=True
    )
    eval_env.unwrapped._add_noise_to_goal = False

    train_dataset = Dataset.create(**train_dataset_dict)
    example_batch = train_dataset.sample(1)

    agent = FBpiSwitchAgent.create(seed=args.seed, ex_batch=example_batch, config=config)
    agent = load_checkpoint(agent, args.checkpoint_dir)


    # Строим граф по датасету
    # Используем все наблюдения из обучающего датасета как вершины графа
    all_observations = np.asarray(train_dataset_dict['observations'])

    # Берём каждое n-е состояние
    subsample_stride = args.subsample_stride
    all_observations = all_observations[::subsample_stride]
    print(f"  Dataset: {len(all_observations)} observations")

    graph_builder = FBGraphBuilder(agent, all_observations)
    graph_builder.build_graph(k=args.k_neighbors, metric=args.graph_metric)

    # Создаём планировщик и обёртку
    # Создаём планировщик
    planner = GraphPlanner(
        graph_builder, agent,
        steps_per_subgoal=25,      # 25 шагов на подцель
        max_subgoals=args.max_subgoals,
    )
    planner_agent = GraphPlannerAgent(
        agent, planner,
        replan_interval=200        # перестройка плана каждые 200 шагов
    )

    dataset_module = importlib.import_module('utils.datasets')
    dataset_class = getattr(dataset_module, config['dataset_class'])

    all_results = {}
    for task_id in range(1, 6):  # 5 задач в OGBench AntMaze
        print(f"\n{'='*60}")
        print(f"Task {task_id}/5")
        print(f"{'='*60}")

        zero_shot_dict = val_dataset_dict if val_dataset_dict else train_dataset_dict
        eval_env.reset(options=dict(task_id=task_id))
        zero_shot_dict = relabel_dataset(args.env_name, eval_env, zero_shot_dict, complex_task_name=None)
        zero_shot_dataset = dataset_class(Dataset.create(**zero_shot_dict), config)

        # Вычисляем z_goal
        num_samples = min(config.get("num_zero_shot_samples", 100_000), zero_shot_dataset.size - 1)
        zero_shot_batch = zero_shot_dataset.sample(
            num_samples, idxs=np.arange(num_samples),
            relabeling=False, augmentation=False,
        )
        inferred_latent = np.asarray(planner_agent.infer_latent(zero_shot_batch))

        # Сбрасываем планировщик перед каждой задачей
        planner_agent.reset_episode()

        # Запуск
        eval_info, trajs, renders = evaluate(
            agent=planner_agent,
            env=eval_env,
            task_id=task_id,
            inferred_latent=inferred_latent,
            num_eval_episodes=args.num_episodes,
            num_video_episodes=0,
            eval_temperature=0.0,
            eval_gaussian=None,
            complex_task_name=None,
        )

        success = eval_info.get('success', 0.0)
        all_results[task_id] = success
        print(f"  Success rate: {success:.3f}")

    print(f"\n{'='*60}")
    print("Graph planner results")
    print(f"{'='*60}")
    for tid, sr in all_results.items():
        print(f"  Task {tid}: {sr:.3f}")
    print(f"  Mean: {np.mean(list(all_results.values())):.3f}")

    output_dir = os.path.join(os.getcwd(), "results")
    os.makedirs(output_dir, exist_ok=True)
    result_path = os.path.join(output_dir, f"graph_planner_seed{args.seed}.pkl")
    with open(result_path, "wb") as f:
        pickle.dump({"results": all_results, "args": vars(args)}, f)
    print(f"\nSaved to: {result_path}")


if __name__ == "__main__":
    main()