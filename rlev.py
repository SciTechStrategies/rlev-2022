import pickle

import click
from sklearn.feature_extraction.text import CountVectorizer

# Minimum document frequency for word features
DEFAULT_WORD_FEATURE_MIN_DF = 250


@click.group()
def cli():
    pass


@cli.command("create-count-vectorizer")
@click.argument("infile", type=click.File("rt"))
@click.argument("outfile", type=click.File("wb"))
@click.option(
    "--min-df",
    type=int,
    help="Minimum document frequency",
    default=DEFAULT_WORD_FEATURE_MIN_DF,
)
def create_count_vectorizer(
    infile,
    outfile,
    min_df,
):
    """Build a CountVectorizer based on input corpus."""
    vectorizer = CountVectorizer(
        min_df=min_df,
        binary=True,
        stop_words="english",
    )
    vectorizer.fit(infile)
    pickle.dump(vectorizer, outfile)


if __name__ == "__main__":
    cli()
