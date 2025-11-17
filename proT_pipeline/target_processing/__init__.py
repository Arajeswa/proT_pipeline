"""
Target processing module for IST (In-System Test) data.

This module processes IST resistance data to generate target dataframes (df_trg.csv)
that are used by the input processing pipeline.
"""

from proT_pipeline.target_processing.main import main
from proT_pipeline.target_processing import modules

__all__ = ['main', 'modules']
