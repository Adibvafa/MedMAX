import operator
import warnings
from typing import *
import traceback
import torch
from dotenv import load_dotenv
from IPython.display import Image
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from transformers import logging
import matplotlib.pyplot as plt
import numpy as np
from interface import create_demo
from medmax.agent import *
from medmax.tools import *
from medmax.utils import *

from medmax.agent import Agent
from medmax.utils import load_prompts_from_file
from medmax.tools import (
    ChestXRayClassifierTool,
    ChestXRayReportGeneratorTool,
    ChestXRaySegmentationTool,
    ImageVisualizerTool,
    XRayPhraseGroundingTool,
    ChestXRayGeneratorTool,
    DicomProcessorTool
)
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI

warnings.filterwarnings("ignore")
logging.set_verbosity_error()
_ = load_dotenv()

def initialize_agent(prompt_file, scratch_dir="afallah", temp_dir="temp"):
    prompts = load_prompts_from_file(prompt_file)
    prompt = prompts["MEDICAL_ASSISTANT"]

    tools_dict = {
        "ChestXRayReportGeneratorTool": ChestXRayReportGeneratorTool(),
        "ChestXRayClassifierTool": ChestXRayClassifierTool(),
        # # # # "MedicalVisualQATool": MedicalVisualQATool(),
        "ChestXRaySegmentationTool": ChestXRaySegmentationTool(),
        "ImageVisualizerTool": ImageVisualizerTool(),
        # "XRayPhraseGroundingTool": XRayPhraseGroundingTool(cache_dir=scratch_dir, temp_dir="temp"),
        # "ChestXRayGeneratorTool": ChestXRayGeneratorTool(model_path=f"{scratch_dir}/roentgen", temp_dir="temp"),
        "DicomProcessorTool": DicomProcessorTool(temp_dir="temp")
    }

    checkpointer = MemorySaver()
    model = ChatOpenAI(model="gpt-4o-2024-08-06", temperature=0, top_p=0.95)
    agent = Agent(
        model,
        tools=list(tools_dict.values()),
        log_tools=True,
        log_dir="logs",
        system_prompt=prompt,
        checkpointer=checkpointer,
    )
    return agent, tools_dict

"""
1. SFT or example database, prompt enhancing to make tool call better
2. show segmentation result overlay on the image
3. use ppx instead of cm2, support dicom
4. look agent benchmark + build benchmark
"""

if __name__ == "__main__":
    agent, tools_dict = initialize_agent("medmax/docs/system_prompts.txt")
    demo = create_demo(agent, tools_dict)
    demo.launch(share=False)

