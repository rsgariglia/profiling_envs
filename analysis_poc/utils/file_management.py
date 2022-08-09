"""File management utility functions
"""
import pickle
from pathlib import Path


def save_pickle(file, filepath_to_save):
    """Save file as pickle file"""
    Path(filepath_to_save).parents[0].mkdir(parents=True, exist_ok=True)
    with open(filepath_to_save, "wb") as out:
        pickle.dump(file, out)


def load_pickle(filepath_to_load):
    """Load pickle file"""
    with open(filepath_to_load, "rb") as inp:
        return pickle.load(inp)
