import sys
import os
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from etl import transform, extract, load

def test_strips_garbage_tokens():
    assert transform("Price is xc2xa3680 today") == "price is 680 today"

def test_lowercases():
    assert transform("BUSINESS News") == "business news"

def test_strips_punctuation():
    assert transform("Wow!!! 10/10, great news.") == "wow 1010 great news"

def test_collapses_extra_whitespace():
    assert transform("too   many    spaces") == "too many spaces"

def make_csv(tmp_path):
    csv_path = tmp_path / "sample.csv"
    pd.DataFrame({
        "data": ["Great match", "Great match", "Terrible loss"],
        "labels": ["sport", "sport", "sport"],
    }).to_csv(csv_path, index=False)
    return str(csv_path)

def test_extract_drops_duplicates(tmp_path):
    df = extract(make_csv(tmp_path))
    assert len(df) == 2

def make_balanced_csv(tmp_path):
    csv_path = tmp_path / "balanced.csv"
    pd.DataFrame({
        "data": [f"article {i}" for i in range(10)],
        "labels": ["sport"] * 5 + ["tech"] * 5,
    }).to_csv(csv_path, index=False)
    return str(csv_path)

def test_load_factorizes_labels(tmp_path):
    X_train, X_test, y_train, y_test, names = load(make_balanced_csv(tmp_path))
    assert set(names) == {"sport", "tech"}
    assert set(y_train.tolist()) | set(y_test.tolist()) == {0, 1}

def test_load_splits_stratified(tmp_path):
    X_train, X_test, y_train, y_test, names = load(make_balanced_csv(tmp_path))
    assert len(X_train) == 8
    assert len(X_test) == 2
