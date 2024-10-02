"""
File: agent.py
-----------------
Define the Agent class. 
"""

from typing import Dict, List, Optional
from tools.config import Tool

import torch
from transformers import pipeline

from agent.utils import load_prompts_from_file


class Agent:
    def __init__(
        self,
        model: str = "meta-llama/Llama-3.2-1B-Instruct",
        tools: Dict[str, Tool] = {},
        system_prompts_file: str = "system_prompts.txt",
        system_prompt_type: str = "MEDICAL_ASSISTANT",
        device: str = "auto",
        torch_dtype: torch.dtype = torch.float16,
    ):
        self.model: str = model
        self.tools: Dict[str, Tool] = tools
        self.device: str = device
        self.torch_dtype: torch.dtype = torch_dtype
        self.system_prompts_file: str = system_prompts_file
        self.system_prompt_type: str = system_prompt_type
        self.pipe: Optional[pipeline] = self.load_pipeline()
        self.system_prompt: str = self.set_system_prompt()
        self.messages: List[Dict[str, str]] = self.initialize_messages()

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
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> str:
        """
        Generate a response from the model and update conversation history.
        """
        self.messages.append({"role": "user", "content": user_message})
        outputs = self.pipe(
            self.messages,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
        response = outputs[0]["generated_text"][-1]
        self.messages.append({"role": "assistant", "content": response})
        return response

    def clear_history(self) -> None:
        """
        Clear the conversation history, keeping only the system message.
        """
        self.messages = self.initialize_messages()

    def initialize_messages(self) -> List[Dict[str, str]]:
        """
        Initialize the messages history with the system prompt.
        """
        return [{"role": "system", "content": self.set_system_prompt()}]

    def set_system_prompt(self) -> str:
        """
        Set or update the system prompt.
        """
        prompts = load_prompts_from_file(self.system_prompts_file)
        return prompts[self.system_prompt_type]
