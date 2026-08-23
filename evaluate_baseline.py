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
from utils.env_utils import make_env_and_datasets
from utils.evaluation import evaluate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--env_name", type=str,
                        default="ogbench-antmaze-medium-navigate-v0")
    parser.add_argument("--num_episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
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



    agent = FBpiSwitchAgent.create(
        seed=args.seed,
        ex_batch=example_batch,
        config=config,
    )


    agent = load_checkpoint(agent, args.checkpoint_dir)


    params = flax.core.unfreeze(agent.network.params)
    for key in sorted(params.keys()):
        print(f"  {key}")

    from utils.env_utils import relabel_dataset

    task_id = 1


    zero_shot_dataset_dict = val_dataset_dict if val_dataset_dict is not None else train_dataset_dict


    eval_env.reset(options=dict(task_id=task_id))


    zero_shot_dataset_dict = relabel_dataset(
        args.env_name,
        eval_env,
        zero_shot_dataset_dict,
        complex_task_name=None,
    )


    dataset_module = importlib.import_module('utils.datasets')
    dataset_class = getattr(dataset_module, config['dataset_class'])
    zero_shot_dataset = dataset_class(Dataset.create(**zero_shot_dataset_dict), config)


    num_samples = min(config.get("num_zero_shot_samples", 100_000), zero_shot_dataset.size - 1)
    zero_shot_batch = zero_shot_dataset.sample(
        num_samples,
        idxs=np.arange(num_samples),
        relabeling=False,
        augmentation=False,
    )
    inferred_latent = np.asarray(agent.infer_latent(zero_shot_batch))
    print(f"z_goal (shape={inferred_latent.shape})")


    print(f"\nEvaluation: {args.num_episodes} episodes, task_id={task_id}...")

    eval_info, trajs, renders = evaluate(
        agent=agent,
        env=eval_env,
        task_id=task_id,
        inferred_latent=inferred_latent,
        num_eval_episodes=args.num_episodes,
        num_video_episodes=0,
        eval_temperature=0.0,
        eval_gaussian=None,
        complex_task_name=None,
    )

    print("\n" + "=" * 60)
    print("Baseline results (single-intention)")
    print("=" * 60)
    for key, value in eval_info.items():
        print(f"  {key}: {value:.4f}")

    # Сохраняем траектории
    output_dir = os.path.join(args.checkpoint_dir, "eval_results")
    os.makedirs(output_dir, exist_ok=True)
    traj_path = os.path.join(output_dir, f"trajs_seed{args.seed}.pkl")
    with open(traj_path, "wb") as f:
        pickle.dump(trajs, f)
    print(f"\nSaved to: {traj_path}")


if __name__ == "__main__":
    main()