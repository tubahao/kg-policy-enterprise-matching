"""Configuration utilities for the knowledge graph generator."""
import os

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def load_config(config_file="config.toml"):
    """Load configuration from TOML file."""
    try:
        with open(config_file, "rb") as f:
            return tomllib.load(f)
    except Exception as e:
        print(f"Error loading config file: {e}")
        return None
