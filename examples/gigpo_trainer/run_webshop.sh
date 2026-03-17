set -x
ENGINE=${1:-vllm}

ulimit -u 65536

export CUDA_VISIBLE_DEVICES=0,3,5
export VLLM_ATTENTION_BACKEND=XFORMERS

# ==============================
# Storage paths (avoid / partition)
# ==============================

export HF_HOME=/data/xinle/hf_cache
export HF_DATASETS_CACHE=/data/xinle/hf_cache/datasets
export TRANSFORMERS_CACHE=/data/xinle/hf_cache/transformers

export RAY_TMPDIR=/data/xinle/ray_tmp
export TMPDIR=/data/xinle/tmp

# ray object spilling (prevents filesystem warning)
export RAY_object_spilling_config='{"type":"filesystem","params":{"directory_path":"/data/xinle/ray_spill"}}'

mkdir -p /data/xinle/hf_cache
mkdir -p /data/xinle/ray_tmp
mkdir -p /data/xinle/ray_spill
mkdir -p /data/xinle/tmp
mkdir -p /data/xinle/verl-agent-checkpoints

# ==============================
# Environment workers
# ==============================

num_cpus_per_env_worker=0.3

# ==============================
# Data size
# ==============================

train_data_size=9
val_data_size=24
group_size=4
mode="mean_norm"

# ==============================
# Data preparation
# ==============================

python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $((val_data_size * 2))

# ==============================
# PPO training
# ==============================

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gigpo \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=3 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    algorithm.gigpo.step_advantage_w=1.0 \
    algorithm.gigpo.mode=$mode \
    env.env_name=Webshop \
    env.seed=0 \
    env.max_steps=15 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_agent_webshop' \
    trainer.experiment_name='gigpo_qwen2.5_7b' \
    trainer.n_gpus_per_node=3 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.default_local_dir='/data/xinle/verl-agent-checkpoints/verl_agent_webshop' \
    trainer.resume_mode=auto \
    trainer.test_freq=20 \
    trainer.total_epochs=150 \
    trainer.val_before_train=True \
    $@