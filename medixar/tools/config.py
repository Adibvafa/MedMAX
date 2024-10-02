"""
File: config.py
-----------------
Define abstract classes for tools.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Dict, Union


class Tool(ABC):
    """
    Abstract base class for tools.
    """

    models: Union[str, List[str]]
    input_types: Dict[str, Any]
    output_type: Any

    def __init__(
        self,
        models: Union[str, List[str]],
        input_types: Dict[str, Any],
        output_type: Any,
    ):
        self.models = [models] if isinstance(models, str) else models
        self.input_types = input_types
        self.output_type = output_type

    @property
    def get_input_types(self) -> Dict[str, Any]:
        """
        Get the expected input types for the tool.
        """
        return self.input_types

    @property
    def get_output_type(self) -> Any:
        """
        Get the expected output type for the tool.
        """
        return self.output_type

    @abstractmethod
    def run(self, **kwargs: Any) -> Any:
        """
        Run the tool with the given inputs.
        """
        pass
