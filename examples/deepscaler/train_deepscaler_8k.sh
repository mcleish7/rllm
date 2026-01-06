set -x

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000

# Find the directory where rllm package is located
RLLM_DIR=$(python3 -c "import rllm; import os; print(os.path.dirname(os.path.dirname(rllm.__file__)))")

MODEL_PATH=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

unset ROCR_VISIBLE_DEVICES
export WANDB_DISABLED=true
wandb offline

# actor_rollout_ref.rollout.n=8 * data.train_batch_size=128 = 1024 generations per train step
# actor_rollout_ref.rollout.val_kwargs.n=8  * data.val_batch_size=30  = 240 generations per val step
# log_prob_micro_batch_size_per_gpu https://github.com/volcengine/verl/blob/2bb42bae6078359c3fdc56ba6c7533e76fc05407/docs/algo/ppo.md#configuration 
# actor_rollout_ref.actor.ppo_mini_batch_size=64 = grad accum = 2

python3 -m examples.deepscaler.train_deepscaler \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.actor.shuffle=False \
    data.shuffle=False \
    data.train_batch_size=128 \
    data.val_batch_size=30 \
    data.max_prompt_length=2048 \
    data.max_response_length=8192 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=30000 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode="async" \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.n=8 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    rllm.mask_truncated_samples=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='rllm-agent' \
    trainer.experiment_name='deepscaler-1.5b-8k-1-node-test' \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.default_hdfs_dir=null \
    rllm.agent.max_steps=1 \
    rllm.stepwise_advantage.enable=False \
    trainer.total_epochs=100

# https://github.com/volcengine/verl/blob/2bb42bae6078359c3fdc56ba6c7533e76fc05407/verl/trainer/ppo/ray_trainer.py#L368C29-L368C45 Hit this

# data_config={'tokenizer': None, 'use_shm': False, 'train_files': '/capstor/scratch/cscs/smcleish/rllm_registry/datasets/deepscaler_math/train_verl.parquet', 'val_files': '/capstor/scratch/cscs/smcleish/rllm_registry/datasets/aime2024/test_verl.parquet', 'train_max_samples': -1, 'val_max_samples': -1, 'prompt_key': 'prompt', 'reward_fn_key': 'data_source', 'max_prompt_length': 2048, 'max_response_length': 8192, 'train_batch_size': 128, 'val_batch_size': 30, 'tool_config_path': None, 'return_raw_input_ids': False, 'return_raw_chat': False, 'return_full_prompt': False, 'shuffle': False, 'seed': None, 'dataloader_num_workers': 8, 'image_patch_size': 14, 'validation_shuffle': False, 'filter_overlong_prompts': False, 'filter_overlong_prompts_workers': 1, 'truncation': 'error', 'image_key': 'images', 'video_key': 'videos', 'trust_remote_code': False, 'custom_cls': {'path': None, 'name': None}, 'return_multi_modal_inputs': False, 'sampler': {'class_path': None, 'class_name': None}, 'datagen': {'path': None, 'name': None}, 'apply_chat_template_kwargs': {}, 'gen_batch_size': 128}