"""
File: agent.py
-----------------
Define the Agent class.
"""

import re
from typing import Dict, List, Optional, Tuple

import torch
from transformers import pipeline

from medixar.tools import Tool
from medixar.utils import load_system_prompt


class Agent:
    def __init__(
        self,
        model: str = "meta-llama/Llama-3.2-1B-Instruct",
        tools: Dict[str, Tool] = {},
        tools_json_path: str = "medixar/docs/tools.json",
        system_prompts_file: str = "medixar/docs/system_prompts.txt",
        system_prompt_type: str = "MEDICAL_ASSISTANT",
        device: str = "auto",
        torch_dtype: torch.dtype = torch.float16,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ):
        self.model: str = model
        self.tools: Dict[str, Tool] = tools
        self.tools_json_path: str = tools_json_path
        self.system_prompts_file: str = system_prompts_file
        self.system_prompt_type: str = system_prompt_type
        self.device: str = device
        self.torch_dtype: torch.dtype = torch_dtype
        self.max_new_tokens: int = max_new_tokens
        self.temperature: float = temperature
        self.top_p: float = top_p
        self.pipe: Optional[pipeline] = self.load_pipeline()
        self.messages: List[Dict[str, str]] = self.initialize_messages()
        self.system_prompt: str = self.get_system_prompt()

    def load_pipeline(self) -> Optional[pipeline]:
        """
        Load model from Hugging Face.
        """
        return pipeline(
            "text-generation",
            model=self.model,
            torch_dtype=self.torch_dtype,
            device_map=self.device,
        )

    def generate(
        self,
        user_message: str,
    ) -> str:
        """
        Generate a response from the model and update conversation history.
        """
        self.messages.append({"role": "user", "content": user_message})
        return self.generate_and_process_response()

    def generate_and_process_response(self) -> str:
        """
        Generate a response from the LLM, process it, and handle tool calls if any.
        """
        response = self.generate_llm_response()
        processed_response, tool_output = self.process_response(response)

        if tool_output:
            assistant_response = response.split("<tool>")[0].strip()
            self.messages.append({"role": "assistant", "content": assistant_response})
            self.messages.append({"role": "tool", "content": tool_output})
            # self.messages.append({"role": "user", "content": ""})
            return self.generate_and_process_response()

        self.messages.append({"role": "assistant", "content": processed_response})
        return processed_response

    def generate_llm_response(self) -> str:
        """
        Generate a response from the LLM.
        """
        outputs = self.pipe(
            self.messages,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        return outputs[0]["generated_text"][-1]["content"]

    def process_response(self, response: str) -> Tuple[str, Optional[str]]:
        """
        Process the response, checking for tool usage and updating the response if necessary.
        """
        tool_output = None
        if "<tool>" in response and "</tool>" in response:
            tool_output = self.execute_tool(response)

        return response, tool_output

    def execute_tool(self, response: str) -> Optional[str]:
        """
        Extract tool call from the response and execute the tool.
        """
        tool_pattern = r"<tool>(.*?)</tool>"
        match = re.search(tool_pattern, response)

        if match:
            tool_content = match.group(1)
            tool_name, tool_args = tool_content.split("(", 1)
            tool_name = tool_name.strip()
            tool_args = tool_args.split(")")[0].strip()
            if tool_name in self.tools:
                return self.tools[tool_name](eval(tool_args))

        return None

    def clear_history(self) -> None:
        """
        Clear the conversation history, keeping only the system message.
        """
        self.messages = self.initialize_messages()

    def initialize_messages(self) -> List[Dict[str, str]]:
        """
        Initialize the messages history with the system prompt.
        """
        return [{"role": "system", "content": self.get_system_prompt()}]

    def get_system_prompt(self) -> str:
        """
        Set the system prompt.
        """
        return load_system_prompt(
            self.system_prompts_file,
            self.system_prompt_type,
            list(self.tools.keys()),
            self.tools_json_path,
        )
