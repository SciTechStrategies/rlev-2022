import pickle
import sys
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


@cli.command("format-input")
@click.argument("infile", type=click.File("rt"))
@click.option("--with-labels", is_flag=True)
def format_input(infile, with_labels):
    n_fields = 9 if with_labels else 8
    for line in infile:
        fields = line.strip().split("\t")
        if len(fields) != n_fields:
            continue
        try:
            if with_labels:
                first_fields = (
                    int(fields[0]),
                    int(fields[1]),
                )
            else:
                first_fields = (int(fields[0]),)

            print(
                *first_fields,
                float(fields[3]),
                float(fields[4]),
                float(fields[5]),
                float(fields[6]),
                fields[7],
                fields[8],
                sep="\t",
            )
        except ValueError:
            continue


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


@cli.command("train-lr-model")
@click.argument("infile", type=click.File("rb"))
@click.argument("outfile", type=click.File("wb"))
def train_lr_model(
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


def _get_combined_model_features(
    *,
    titles,
    ref_probs,
    abstracts,
    title_vectorizer,
    abstr_vectorizer,
    rlev_priors,
    word_feature_model,
    min_word_features,
):
    ref_probs = np.matrix([[np.float64(x) for x in line] for line in ref_probs])

    title_features = title_vectorizer.transform(titles)
    abstr_features = abstr_vectorizer.transform(abstracts)
    word_features = sparse.hstack((title_features, abstr_features))

    row_sums = word_features.sum(axis=1)
    wc_mask = np.less(row_sums, min_word_features).A1
    word_probs = word_feature_model.predict_proba(word_features)

    # If a given row has fewer than `min_word_features`
    # word features, then use priors instead of results
    # from word feature model.
    word_probs[wc_mask] = rlev_priors

    features = np.hstack((ref_probs, word_probs))
    return features


@cli.command("create-combined-model-inputs")
@click.argument("infile", type=click.File("rt"))
@click.argument("outfile", type=click.File("wb"))
@click.option("--title-vectorizer", type=click.File("rb"), required=True)
@click.option("--abstr-vectorizer", type=click.File("rb"), required=True)
@click.option("--word-feature-model", type=click.File("rb"), required=True)
@click.option("--rlev-priors", type=click.File("rb"), required=True)
@click.option(
    "--min-word-features",
    type=int,
    default=DEFAULT_WORD_FEATURE_MIN_FEATURES,
    help="Minimum number of word features for a document to be "
    "included in output matrix/labels.",
)
def create_combined_model_inputs(
    infile,
    outfile,
    title_vectorizer,
    abstr_vectorizer,
    word_feature_model,
    rlev_priors,
    min_word_features,
):
    """Create the input matrix and label array for the combined model."""

    title_vectorizer = pickle.load(title_vectorizer)
    abstr_vectorizer = pickle.load(abstr_vectorizer)
    word_feature_model = pickle.load(word_feature_model)
    rlev_priors = pickle.load(rlev_priors)

    Y = None
    X = None

    line_chunks = chunks(infile, 100000)
    for chunk in line_chunks:

        lines = [line.strip().split("\t") for line in chunk]
        # Only select rows where we can extract 7 fields.
        # rlev, ref-prob 1-4, title, abstract
        lines = [f for f in lines if len(f) == 7]

        if not lines:
            continue

        features = _get_combined_model_features(
            ref_probs=[line[1:5] for line in lines],
            titles=[line[5] for line in lines],
            abstracts=[line[6] for line in lines],
            title_vectorizer=title_vectorizer,
            abstr_vectorizer=abstr_vectorizer,
            word_feature_model=word_feature_model,
            min_word_features=min_word_features,
            rlev_priors=rlev_priors,
        )

        # Subtract 1 since input rlevs are 1-indexed
        rlevs = np.array([int(f[0]) - 1 for f in lines])

        if X is None:
            X = features
        else:
            X = np.vstack((X, features))

        if Y is None:
            Y = rlevs
        else:
            Y = np.hstack((Y, rlevs))

    print("X:", X.shape, type(X))
    print("Y:", Y.shape, type(Y))
    pickle.dump((X, Y), outfile)


@cli.command("get-combined-model-predictions")
@click.argument("infile", type=click.File("rt"))
@click.option("--title-vectorizer", type=click.File("rb"), required=True)
@click.option("--abstr-vectorizer", type=click.File("rb"), required=True)
@click.option("--word-feature-model", type=click.File("rb"), required=True)
@click.option("--rlev-priors", type=click.File("rb"), required=True)
@click.option(
    "--min-word-features",
    type=int,
    default=DEFAULT_WORD_FEATURE_MIN_FEATURES,
    help="Minimum number of word features for a document to be "
    "included in output matrix/labels.",
)
@click.option("--combined-model", type=click.File("rb"), required=True)
@click.option("--with-labels", is_flag=True)
def get_combined_model_predictions(
    infile,
    title_vectorizer,
    abstr_vectorizer,
    word_feature_model,
    rlev_priors,
    min_word_features,
    combined_model,
    with_labels,
):
    """Get combined model probabilities."""

    title_vectorizer = pickle.load(title_vectorizer)
    abstr_vectorizer = pickle.load(abstr_vectorizer)
    word_feature_model = pickle.load(word_feature_model)
    rlev_priors = pickle.load(rlev_priors)
    combined_model = pickle.load(combined_model)

    line_chunks = chunks(infile, 100000)

    n_fields = 8 if with_labels else 7
    pre = 2 if with_labels else 1

    for chunk in line_chunks:

        lines = [line.strip().split("\t") for line in chunk]
        # Only select rows where we can extract `n_fields` fields.
        # id, rlev (if with_labels is True), ref-prob 1-4, title, abstract
        lines = [f for f in lines if len(f) == n_fields]

        if not lines:
            continue

        features = _get_combined_model_features(
            ref_probs=[line[pre : pre + 4] for line in lines],
            titles=[line[pre + 4] for line in lines],
            abstracts=[line[pre + 5] for line in lines],
            title_vectorizer=title_vectorizer,
            abstr_vectorizer=abstr_vectorizer,
            word_feature_model=word_feature_model,
            min_word_features=min_word_features,
            rlev_priors=rlev_priors,
        )
        id_rlev = np.matrix([line[0:pre] for line in lines])
        probs = combined_model.predict_proba(features)
        results = np.hstack((id_rlev, probs))

        np.savetxt(sys.stdout, results, fmt="%s", delimiter="\t")


if __name__ == "__main__":
    cli()
