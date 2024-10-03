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

    api_name: str
    input_args: Dict[str, Any]
    output_type: Any
    api: Any

    def __init__(
        self,
        api_name: str,
        input_args: Dict[str, Any],
        output_type: Any,
    ):
        self.api_name = api_name
        self.input_args = input_args
        self.output_type = output_type
        self.api = None

    @property
    def get_input_args(self) -> Dict[str, Any]:
        """
        Get the expected input args for the tool.
        """
        return self.input_args

    @property
    def get_output_type(self) -> Any:
        """
        Get the expected output type for the tool.
        """
        return self.output_type

    @abstractmethod
    def setup_api(self):
        """
        Setup the API for the tool.
        """
        pass

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Run the tool with the given inputs.
        """
        pass


class WebSearchTool(Tool):
    """
    Tool for web search.
    """

    def __init__(self):
        super().__init__(
            api_name="web_search",
            input_args={"query": str},
            output_type=str,
        )
        self.setup_api()

    def setup_api(self):
        """
        Setup the API for the tool.
        """
        self.api = lambda query: f"Syaro is best girl given query: `{query}`"

    def __call__(self, query: str, *args: Any, **kwargs: Any) -> str:
        """
        Run the tool with the given inputs.
        """
        return self.api(query)
