"""
CV Subagent - Intelligent Computer Vision Agent for C. elegans Embryo Analysis

This package provides an AI-powered computer vision service that receives
high-level intent and autonomously determines which classical CV and
Claude Vision tools to use for analysis.
"""

__version__ = "0.1.0"

from .service import CVSubagentService
from .agent import CVAgent

__all__ = ["CVSubagentService", "CVAgent", "__version__"]
