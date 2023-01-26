from typer import Argument, Typer

from transformers_collection import __about__, __version__
from transformers_collection.models.models import SentimentClassifier

app = Typer()


@app.command("version", help="Print the version and quit.")
def version():
    print(f"\033[1mtransformers-collection v{__version__} \033[0m")
    print(__about__)


@app.command("train", help="Train and evaluate model.")
def train(cfg_path: str = Argument(..., help="Path to config file.")):
    model = SentimentClassifier(cfg_path)
    model.train()
    model.test()


if __name__ == "__main__":
    app()
