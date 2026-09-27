from config import Config
from experiment_config import experiment_config
from train_evaluate import run_experiment
from utils import save_results


def run_all_modes(config: Config) -> None:
    for feature_mode in config.feature_modes:
        for input_mode in config.input_modes:
            print("\n" + "=" * 80)
            print(
                f"Running FEATURE_MODE={feature_mode} | "
                f"INPUT_MODE={input_mode}"
            )
            print(
                f"TARGET={config.target_label} | "
                f"CLASSIFICATION={config.classification}"
            )
            print(f"MODELS={config.models_to_run}")
            print("=" * 80)

            summary_df, subjects_df = run_experiment(
                config=config,
                feature_mode=feature_mode,
                input_mode=input_mode,
            )

            print("\n===== FINAL RESULTS =====\n")
            print(summary_df)

            save_results(
                summary_df=summary_df,
                subjects_df=subjects_df,
                config=config,
                input_type=feature_mode,
                input_mode=input_mode,
            )


def main() -> None:
    run_all_modes(config=experiment_config)


if __name__ == "__main__":
    main()
