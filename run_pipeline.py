import click
from pipelines.training_pipeline import ml_pipeline


@click.command
def main():
    """
    Run the ML pipeline
    """

    run = ml_pipeline()


if __name__ == "__main__":
    main()
