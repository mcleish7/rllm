import hydra

from rllm.agents.math_agent import MathAgent
from rllm.data.dataset import DatasetRegistry
from rllm.environments.base.single_turn_env import SingleTurnEnvironment
from rllm.rewards.reward_fn import math_reward_fn
from rllm.trainer.agent_trainer import AgentTrainer


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    train_name = config.get("dataset", {}).get("train_name", "deepscaler_math")
    train_split = config.get("dataset", {}).get("train_split", "train")
    val_name = config.get("dataset", {}).get("val_name", "aime2024")
    val_split = config.get("dataset", {}).get("val_split", "test")

    train_dataset = DatasetRegistry.load_dataset(train_name, train_split)
    test_dataset = DatasetRegistry.load_dataset(val_name, val_split)

    env_args = {"reward_fn": math_reward_fn}

    trainer = AgentTrainer(
        agent_class=MathAgent,
        agent_args={},
        env_args=env_args,
        env_class=SingleTurnEnvironment,
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
