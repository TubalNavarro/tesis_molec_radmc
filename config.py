import json
from pathlib import Path


def load_config(config_file):
    """
    Load project configuration from a JSON file.

    Parameters
    ----------
    config_file : str or Path
        Path to the JSON configuration file.

    Returns
    -------
    config : dict
        Configuration dictionary.
    """

    config_file = Path(config_file)

    if not config_file.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_file}"
        )

    with open(config_file, "r") as f:
        config = json.load(f)

    return config