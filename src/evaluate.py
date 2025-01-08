import tyro

from config import EvaluateConfig

if __name__ == "__main__":
    config = tyro.cli(EvaluateConfig)
