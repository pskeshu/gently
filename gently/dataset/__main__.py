"""
Entry point for running dataset CLI as a module.

Usage:
    python -m gently.dataset aggregate
    python -m gently.dataset stats
    python -m gently.dataset serve
"""

from .cli import main

if __name__ == "__main__":
    main()
