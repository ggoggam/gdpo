import tyro

from config import CompareConfig, DiversityConfig
from evaluate import DiversityEvaluator, WinRateEvaluator

if __name__ == "__main__":
    config = tyro.extras.overridable_config_cli(
        {
            "diversity": (
                "Diversity",
                DiversityConfig(),
            ),
            "win": (
                "Win Rate",
                CompareConfig(),
            ),
        }
    )

    if config.metric == "diversity":
        evaluator_cls = DiversityEvaluator
    elif config.metric == "win":
        evaluator_cls = WinRateEvaluator

    evaluator = evaluator_cls(config)
    metric = evaluator.evaluate()
    print(metric)
