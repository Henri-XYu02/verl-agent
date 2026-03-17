#!/bin/bash
set -x

export CUDA_VISIBLE_DEVICES=1,3,6
export VLLM_ATTENTION_BACKEND=XFORMERS
export TRANSFORMERS_CACHE=/data/xinle/hf_cache/transformers
export HF_HOME=/data/xinle/hf_cache
export FIRE_BENCH_IMAGE=fire-bench-sandbox:latest

ENGINE=${1:-vllm}

ulimit -u 65536

# CPU allocation for FireBench Docker workers
num_cpus_per_env_worker=1

# ---- dataset sizes ----
train_data_size=3
val_data_size=5

# trajectories per task
group_size=2

mode="mean_norm"

TRAIN_FILE="$HOME/data/fire_bench/train.parquet"
VAL_FILE="$HOME/data/fire_bench/val.parquet"

# ---------------------------
# Build dataset if needed
# ---------------------------
if [ ! -f "$TRAIN_FILE" ]; then
    echo "Building FIRE-Bench dataset..."
    python3 tools/build_firebench_dataset.py \
        --output_dir "$HOME/data/fire_bench"
fi

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gigpo \
\
    data.train_files=$TRAIN_FILE \
    data.val_files=$VAL_FILE \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
\
    data.return_raw_chat=True \
    data.max_prompt_length=8192 \
    data.max_response_length=2048 \
    data.truncation=left \
\
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
\
    actor_rollout_ref.actor.optim.lr=1e-6 \
\
    actor_rollout_ref.actor.ppo_mini_batch_size=6 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
\
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
\
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
\
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
\
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
\
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
\
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
\
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
\
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
\
    algorithm.gamma=1.0 \
    algorithm.gigpo.step_advantage_w=1.0 \
    algorithm.gigpo.mode=$mode \
\
    env.env_name=fire_bench \
    env.seed=0 \
    env.max_steps=20 \
    env.history_length=5 \
\
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
\
    trainer.critic_warmup=0 \
\
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_agent_fire_bench' \
    trainer.experiment_name='gigpo_qwen2.5_7b' \
\
    trainer.n_gpus_per_node=3 \
    trainer.nnodes=1 \
\
    trainer.save_freq=10 \
    trainer.test_freq=5 \
\
    trainer.default_local_dir=/data/xinle/checkpoints/verl_agent_fire_bench \
    trainer.resume_mode=auto \
\
    trainer.total_epochs=1 \
    trainer.val_before_train=True \
\
    "$@"