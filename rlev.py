import pickle
from collections import Counter

import click
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# Minimum document frequency for word features
DEFAULT_WORD_FEATURE_MIN_DF = 250

# Minimum number of word features for document
# to be included when building word feature model
DEFAULT_WORD_FEATURE_MIN_FEATURES = 2


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


def chunks(items, n):
    curr_chunk = []
    for item in items:
        curr_chunk.append(item)
        if len(curr_chunk) == n:
            yield curr_chunk
            curr_chunk = []
    if curr_chunk:
        yield curr_chunk


@cli.command("create-word-feature-model-inputs")
@click.argument("infile", type=click.File("rt"))
@click.argument("outfile", type=click.File("wb"))
@click.option("--title-vectorizer", type=click.File("rb"), required=True)
@click.option("--abstr-vectorizer", type=click.File("rb"), required=True)
@click.option(
    "--min-word-features",
    type=int,
    default=DEFAULT_WORD_FEATURE_MIN_FEATURES,
    help="Minimum number of word features for a document to be "
    "included in output matrix/labels.",
)
def create_word_feature_model_inputs(
    infile,
    outfile,
    title_vectorizer,
    abstr_vectorizer,
    min_word_features,
):
    """Create the input matrix and label array for a word feature model."""

    title_vectorizer = pickle.load(title_vectorizer)
    abstr_vectorizer = pickle.load(abstr_vectorizer)

    Y = None
    X = None

    line_chunks = chunks(infile, 100000)
    for chunk in line_chunks:
        fields = [line.strip().split("\t") for line in chunk]

        # Only select rows where we can extract 3 fields.
        fields = [f for f in fields if len(f) == 3]

        title_features = title_vectorizer.transform(f[1] for f in fields)
        abstr_features = abstr_vectorizer.transform(f[2] for f in fields)
        features = sparse.hstack((title_features, abstr_features))

        # Subtract 1 since input rlevs are 1-indexed
        rlevs = np.array([int(f[0]) - 1 for f in fields])

        row_sums = features.sum(axis=1)
        wc_mask = np.greater_equal(row_sums, min_word_features).A1

        features = features[wc_mask, :]
        rlevs = rlevs[wc_mask]

        if X is None:
            X = features
        else:
            X = sparse.vstack((X, features))

        if Y is None:
            Y = rlevs
        else:
            Y = np.hstack((Y, rlevs))

    print("X:", X.shape, type(X))
    print("Y:", Y.shape, type(Y))
    pickle.dump((X, Y), outfile)


@cli.command("train-word-feature-model")
@click.argument("infile", type=click.File("rb"))
@click.argument("outfile", type=click.File("wb"))
def train_word_feature_model(
    infile,
    outfile,
):
    """Train word feature model."""
    X, Y = pickle.load(infile)
    model = LogisticRegression(
        penalty="l2",
        C=1e5,
    )
    model.fit(X, Y)
    pickle.dump(model, outfile)


@cli.command("get-rlev-priors")
@click.argument("infile", type=click.File("rt"))
@click.argument("outfile", type=click.File("wb"))
def get_rlev_priors(
    infile,
    outfile,
):
    """Create the input matrix and label array for a word feature model."""
    # 1-indexed input
    counts = Counter(int(line.strip()) - 1 for line in infile)
    total_count = sum(counts.values())
    priors = [
        counts.get(rlev, 0) / total_count for rlev in range(0, max(counts.keys()) + 1)
    ]
    pickle.dump(priors, outfile)


if __name__ == "__main__":
    cli()
