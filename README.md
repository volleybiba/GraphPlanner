## Установка

```bash
pip install -r requirements.txt
pip install ogbench
```

## Запуск бейзлайна

```bash
python evaluate_baseline.py \
    --checkpoint_dir checkpoints/baseline \
    --env_name ogbench-antmaze-medium-navigate-v0 \
    --num_episodes 20 --seed 0
```

## Запуск графового планировщика

```bash
python evaluate_graph_planner.py \
    --checkpoint_dir checkpoints/baseline \
    --env_name ogbench-antmaze-medium-navigate-v0 \
    --num_episodes 20 --seed 0 \
    --k_neighbors 10 --max_subgoals 8 --subsample_stride 5
```


## Если JAX выдает ошибку и принудительно запускает программу на CPU
```bash
SITE=/home/maxim/switching-successor-measures/ssm/lib/python3.12/site-packages

export LD_LIBRARY_PATH="$SITE/nvidia/cusparse/lib:$SITE/nvidia/cublas/lib:$SITE/nvidia/cudnn/lib:$SITE/nvidia/cufft/lib:$SITE/nvidia/cusolver/lib:$SITE/nvidia/nccl/lib:$SITE/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH"
```

