"""Split the tickets once and draw small stratified samples."""

from sklearn.model_selection import train_test_split

from tickets_data import make_tickets

SPLIT_SEED = 2026
SAMPLE_SIZES = (50, 150, 400)

def split_tickets(frame, seed=SPLIT_SEED):
    """Return train, validation and test rows (60/20/20), stratified by label."""
    train, rest = train_test_split(
        frame, test_size=0.4, stratify=frame["label"], random_state=seed
    )
    validation, test = train_test_split(
        rest, test_size=0.5, stratify=rest["label"], random_state=seed
    )
    return (
        train.reset_index(drop=True),
        validation.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def load_split(seed=SPLIT_SEED):
    """Make the tickets with default arguments and split them."""
    return split_tickets(make_tickets(), seed)


def sample_training(train, n, seed):
    """Draw a stratified sample of n rows from the training rows only."""
    if n > len(train):
        raise ValueError(f"Cannot draw {n} rows from {len(train)} training rows.")
    if n == len(train):
        return train.copy()
    sample, _ = train_test_split(
        train, train_size=n, stratify=train["label"], random_state=seed
    )
    return sample.reset_index(drop=True)