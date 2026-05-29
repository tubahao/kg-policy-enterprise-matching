"""
Session 2 — Knowledge Graph Triple Extraction (Refactored).

Ontology-constrained extraction pipeline:
  - deterministic_edges.py  : zero-LLM edge generation
  - llm_classifier.py       : multi-label sub-industry classification
  - verifier.py             : validation, merge, ontology generation
"""

from src.extraction.llm import call_llm, extract_json_from_text
from src.extraction.config import load_config

__version__ = "2.0"
