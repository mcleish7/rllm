from datasets import load_dataset

from rllm.data.dataset import DatasetRegistry


def prepare_math_data():
    train_dataset = load_dataset("agentica-org/DeepScaleR-Preview-Dataset", split="train")
    test_dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")

    easy_dataset = load_dataset("smcleish/error_at_k_saved_start_0_end_20000_num_completions_10", split="easy")
    hard_dataset = load_dataset("smcleish/error_at_k_saved_start_0_end_20000_num_completions_10", split="hard")
    random_dataset = load_dataset("smcleish/error_at_k_saved_start_0_end_20000_num_completions_10", split="random")

    def preprocess_fn(example, idx):
        return {
            "question": example["problem"],
            "ground_truth": example["answer"],
            "data_source": "math",
        }

    train_dataset = train_dataset.map(preprocess_fn, with_indices=True)
    test_dataset = test_dataset.map(preprocess_fn, with_indices=True)

    easy_dataset = easy_dataset.map(preprocess_fn, with_indices=True)
    hard_dataset = hard_dataset.map(preprocess_fn, with_indices=True)
    random_dataset = random_dataset.map(preprocess_fn, with_indices=True)

    train_dataset = DatasetRegistry.register_dataset("deepscaler_math", train_dataset, "train")
    test_dataset = DatasetRegistry.register_dataset("aime2024", test_dataset, "test")

    easy_dataset = DatasetRegistry.register_dataset("deepscaler_math_error_at_k_saved_start_0_end_20000_num_completions_10_easy", easy_dataset, "train")
    hard_dataset = DatasetRegistry.register_dataset("deepscaler_math_error_at_k_saved_start_0_end_20000_num_completions_10_hard", hard_dataset, "train")
    random_dataset = DatasetRegistry.register_dataset("deepscaler_math_error_at_k_saved_start_0_end_20000_num_completions_10_random", random_dataset, "train")
    print(easy_dataset.get_data_path())
    print(hard_dataset.get_data_path())
    print(random_dataset.get_data_path())

    return train_dataset, test_dataset


if __name__ == "__main__":
    train_dataset, test_dataset = prepare_math_data()
    print(train_dataset.get_data_path())
    print(test_dataset.get_data_path())
