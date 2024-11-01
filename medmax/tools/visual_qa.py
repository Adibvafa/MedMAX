from typing import Any, Dict, Optional, Tuple, Type
from pydantic import BaseModel, Field

import torch

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool

from PIL import Image


from medmax.llava.conversation import conv_templates
from medmax.llava.model.builder import load_pretrained_model
from medmax.llava.mm_utils import tokenizer_image_token, process_images
from medmax.llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)


class MedicalVisualQAInput(BaseModel):
    """Input for the Medical Visual QA tool."""

    question: str = Field(..., description="The question to ask about the medical image")
    image_path: Optional[str] = Field(None, description="Path to the medical image file (optional)")


class MedicalVisualQATool(BaseTool):
    """Tool that performs medical visual question answering using a pre-trained model.

    This tool uses a large language model fine-tuned on medical images to answer
    questions about medical images. It can handle both image-based questions and
    general medical questions without images.
    """

    name: str = "medical_visual_qa"
    description: str = (
        "A tool that answers medical questions about images using a pre-trained model. "
        "It can also answer general medical questions without images. "
        "Input should be a question and optionally a path to a medical image file."
    )
    args_schema: Type[BaseModel] = MedicalVisualQAInput
    tokenizer: Any = None
    model: Any = None
    image_processor: Any = None
    context_len: int = 200000

    def __init__(self, model_path: str = "/model-weights/llava-med-v1.5-mistral-7b", **kwargs):
        super().__init__()
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path, None, model_path, **kwargs
        )
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def _process_input(
        self, question: str, image_path: Optional[str] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.model.config.mm_use_im_start_end:
            question = (
                DEFAULT_IM_START_TOKEN
                + DEFAULT_IMAGE_TOKEN
                + DEFAULT_IM_END_TOKEN
                + "\n"
                + question
            )
        else:
            question = DEFAULT_IMAGE_TOKEN + "\n" + question

        conv = conv_templates["vicuna_v1"].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = (
            tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )

        image_tensor = None
        if image_path:
            image = Image.open(image_path)
            image_tensor = process_images([image], self.image_processor, self.model.config)[0]
            image_tensor = image_tensor.unsqueeze(0).half().cuda()

        return input_ids, image_tensor

    def _run(
        self,
        question: str,
        image_path: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[str, Dict]:
        """Answer a medical question, optionally based on an input image.

        Args:
            question (str): The medical question to answer.
            image_path (Optional[str]): The path to the medical image file (if applicable).
            run_manager (Optional[CallbackManagerForToolRun]): The callback manager for the tool run.

        Returns:
            Tuple[str, Dict]: A tuple containing the model's answer and any additional metadata.

        Raises:
            Exception: If there's an error processing the input or generating the answer.
        """
        try:
            input_ids, image_tensor = self._process_input(question, image_path)

            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=False,
                    temperature=0.2,
                    max_new_tokens=500,
                    use_cache=True,
                )

            output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            metadata = {
                "question": question,
                "image_path": image_path,
                "analysis_status": "completed",
            }
            return output, metadata
        except Exception as e:
            return f"Error generating answer: {str(e)}", {
                "question": question,
                "image_path": image_path,
                "analysis_status": "failed",
            }

    async def _arun(
        self,
        question: str,
        image_path: Optional[str] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Tuple[str, Dict]:
        """Asynchronously answer a medical question, optionally based on an input image.

        This method currently calls the synchronous version, as the model inference
        is not inherently asynchronous. For true asynchronous behavior, consider
        using a separate thread or process.

        Args:
            question (str): The medical question to answer.
            image_path (Optional[str]): The path to the medical image file (if applicable).
            run_manager (Optional[AsyncCallbackManagerForToolRun]): The async callback manager for the tool run.

        Returns:
            Tuple[str, Dict]: A tuple containing the model's answer and any additional metadata.

        Raises:
            Exception: If there's an error processing the input or generating the answer.
        """
        return self._run(question, image_path)
