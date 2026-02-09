"""
Data analysis module for proT_pipeline.

This module provides tools for analyzing the relationship between
input parameters and targets, including HSIC-based feature selection.
"""

from .hsic import run_hsic_analysis, compute_baseline_hsic

__all__ = ['run_hsic_analysis', 'compute_baseline_hsic']
