"""Base class for screening steps."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List

from ..config import PipelineConfig
from ..calculators.base import CalculatorBase

logger = logging.getLogger(__name__)


class BaseStep(ABC):
    """Base class for all screening steps."""

    def __init__(self, config: PipelineConfig, calculator: CalculatorBase, output_dir: Path):
        """Initialize step.

        Args:
            config: Pipeline configuration
            calculator: ML potential calculator
            output_dir: Output directory for this run
        """
        self.config = config
        self.calculator = calculator
        self.output_dir = output_dir

    @abstractmethod
    def run(self, input_data: Any) -> Any:
        """Execute the screening step.

        Args:
            input_data: Input from previous step

        Returns:
            Output data for next step
        """
        ...
