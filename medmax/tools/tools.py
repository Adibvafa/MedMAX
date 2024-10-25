from typing import Any, Dict, Optional, Tuple, Type, Union
from pydantic import BaseModel, Field

import skimage.io
import torch
import torchvision
import torchxrayvision as xrv

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool

from pydantic import BaseModel, Field
import torch
from PIL import Image

from transformers import (
    BertTokenizer,
    ViTImageProcessor,
    VisionEncoderDecoderModel,
    GenerationConfig
)

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

    def __init__(self, model_path: str = "microsoft/llava-med-v1.5-mistral-7b", **kwargs):
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


class ChestXRayInput(BaseModel):
    """Input for chest X-ray analysis tools."""

    image_path: str = Field(..., description="Path to the radiology image file")


class ChestXRayClassifierTool(BaseTool):
    """Tool that classifies chest X-ray images for multiple pathologies.

    This tool uses a pre-trained DenseNet model to analyze chest X-ray images and
    predict the likelihood of various pathologies. The model can classify the following 18 conditions:

    Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema,
    Enlarged Cardiomediastinum, Fibrosis, Fracture, Hernia, Infiltration,
    Lung Lesion, Lung Opacity, Mass, Nodule, Pleural Thickening, Pneumonia, Pneumothorax

    The output values represent the probability (from 0 to 1) of each condition being present in the image.
    A higher value indicates a higher likelihood of the condition being present.
    """

    name: str = "chest_xray_classifier"
    description: str = (
        "A tool that analyzes chest X-ray images and classifies them for 18 different pathologies. "
        "Input should be the path to a chest X-ray image file. "
        "Output is a dictionary of pathologies and their predicted probabilities (0 to 1). "
        "Pathologies include: Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema, "
        "Enlarged Cardiomediastinum, Fibrosis, Fracture, Hernia, Infiltration, Lung Lesion, "
        "Lung Opacity, Mass, Nodule, Pleural Thickening, Pneumonia, and Pneumothorax. "
        "Higher values indicate a higher likelihood of the condition being present."
    )
    args_schema: Type[BaseModel] = ChestXRayInput
    model: xrv.models.DenseNet = None
    transform: torchvision.transforms.Compose = None

    def __init__(self, model_name: str = "densenet121-res224-all"):
        super().__init__()
        self.model = xrv.models.DenseNet(weights=model_name)
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop()])

    def _process_image(self, image_path: str) -> torch.Tensor:
        """
        Process the input chest X-ray image for model inference.

        This method loads the image, normalizes it, applies necessary transformations,
        and prepares it as a torch.Tensor for model input.

        Args:
            image_path (str): The file path to the chest X-ray image.

        Returns:
            torch.Tensor: A processed image tensor ready for model inference.

        Raises:
            FileNotFoundError: If the specified image file does not exist.
            ValueError: If the image cannot be properly loaded or processed.
        """
        img = skimage.io.imread(image_path)
        img = xrv.datasets.normalize(img, 255)

        if len(img.shape) > 2:
            img = img[:, :, 0]

        img = img[None, :, :]
        img = self.transform(img)
        img = torch.from_numpy(img).unsqueeze(0)

        if torch.cuda.is_available():
            img = img.cuda()

        return img

    def _run(
        self,
        image_path: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, float], Dict]:
        """Classify the chest X-ray image for multiple pathologies.

        Args:
            image_path (str): The path to the chest X-ray image file.
            run_manager (Optional[CallbackManagerForToolRun]): The callback manager for the tool run.

        Returns:
            Tuple[Dict[str, float], Dict]: A tuple containing the classification results
                                           (pathologies and their probabilities from 0 to 1)
                                           and any additional metadata.

        Raises:
            Exception: If there's an error processing the image or during classification.
        """
        try:
            img = self._process_image(image_path)
            with torch.no_grad():
                preds = self.model(img).cpu()[0]
            output = dict(zip(xrv.datasets.default_pathologies, preds.numpy()))
            metadata = {
                "image_path": image_path,
                "analysis_status": "completed",
                "note": "Probabilities range from 0 to 1, with higher values indicating higher likelihood of the condition.",
            }
            return output, metadata
        except Exception as e:
            return {"error": str(e)}, {
                "image_path": image_path,
                "analysis_status": "failed",
            }

    async def _arun(
        self,
        image_path: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, float], Dict]:
        """Asynchronously classify the chest X-ray image for multiple pathologies.

        This method currently calls the synchronous version, as the model inference
        is not inherently asynchronous. For true asynchronous behavior, consider
        using a separate thread or process.

        Args:
            image_path (str): The path to the chest X-ray image file.
            run_manager (Optional[AsyncCallbackManagerForToolRun]): The async callback manager for the tool run.

        Returns:
            Tuple[Dict[str, float], Dict]: A tuple containing the classification results
                                           (pathologies and their probabilities from 0 to 1)
                                           and any additional metadata.

        Raises:
            Exception: If there's an error processing the image or during classification.
        """
        return self._run(image_path)


class ChestXRayReportGeneratorTool(BaseTool):
    """Tool that generates comprehensive chest X-ray reports with both findings and impressions.
    
    This tool uses two Vision-Encoder-Decoder models (ViT-BERT) trained on CheXpert 
    and MIMIC-CXR datasets to generate structured radiology reports. It automatically 
    generates both detailed findings and impression summaries for each chest X-ray,
    following standard radiological reporting format.

    The tool uses:
    - Findings model: Generates detailed observations of all visible structures
    - Impression model: Provides concise clinical interpretation and key diagnoses
    """

    name: str = "chest_xray_report_generator"
    description: str = (
        "A tool that analyzes chest X-ray images and generates comprehensive radiology reports "
        "containing both detailed findings and impression summaries. Input should be the path "
        "to a chest X-ray image file. Output is a structured report with both detailed "
        "observations and key clinical conclusions."
    )
    args_schema: Type[BaseModel] = ChestXRayInput
    findings_model: VisionEncoderDecoderModel = None
    impression_model: VisionEncoderDecoderModel = None
    findings_tokenizer: BertTokenizer = None
    impression_tokenizer: BertTokenizer = None
    findings_processor: ViTImageProcessor = None
    impression_processor: ViTImageProcessor = None
    generation_args: Dict[str, Any] = None

    def __init__(self):
        """Initialize the ChestXRayReportGeneratorTool with both findings and impression models."""
        super().__init__()
        
        # Initialize findings model
        self.findings_model = VisionEncoderDecoderModel.from_pretrained(
            "IAMJB/chexpert-mimic-cxr-findings-baseline"
        ).eval()
        self.findings_tokenizer = BertTokenizer.from_pretrained(
            "IAMJB/chexpert-mimic-cxr-findings-baseline"
        )
        self.findings_processor = ViTImageProcessor.from_pretrained(
            "IAMJB/chexpert-mimic-cxr-findings-baseline"
        )

        # Initialize impression model
        self.impression_model = VisionEncoderDecoderModel.from_pretrained(
            "IAMJB/chexpert-mimic-cxr-impression-baseline"
        ).eval()
        self.impression_tokenizer = BertTokenizer.from_pretrained(
            "IAMJB/chexpert-mimic-cxr-impression-baseline"
        )
        self.impression_processor = ViTImageProcessor.from_pretrained(
            "IAMJB/chexpert-mimic-cxr-impression-baseline"
        )
        
        # Move models to GPU if available
        if torch.cuda.is_available():
            self.findings_model = self.findings_model.cuda()
            self.impression_model = self.impression_model.cuda()
        
        # Default generation arguments
        self.generation_args = {
            "num_return_sequences": 1,
            "max_length": 128,
            "use_cache": True,
            "beam_width": 2,
        }

    def _process_image(self, image_path: str, processor: ViTImageProcessor, model: VisionEncoderDecoderModel) -> torch.Tensor:
        """Process the input image for a specific model.
        
        Args:
            image_path (str): Path to the input image.
            processor: Image processor for the specific model.
            model: The model to process the image for.
            
        Returns:
            torch.Tensor: Processed image tensor ready for model input.
        """
        image = Image.open(image_path).convert('RGB')
        pixel_values = processor(image, return_tensors="pt").pixel_values
        
        expected_size = model.config.encoder.image_size
        actual_size = pixel_values.shape[-1]
        
        if expected_size != actual_size:
            pixel_values = torch.nn.functional.interpolate(
                pixel_values,
                size=(expected_size, expected_size),
                mode='bilinear',
                align_corners=False
            )
        
        if torch.cuda.is_available():
            pixel_values = pixel_values.cuda()
            
        return pixel_values

    def _generate_report_section(self, pixel_values: torch.Tensor, model: VisionEncoderDecoderModel, 
                               tokenizer: BertTokenizer) -> str:
        """Generate a report section using the specified model.
        
        Args:
            pixel_values: Processed image tensor.
            model: The model to use for generation.
            tokenizer: The tokenizer for the model.
            
        Returns:
            str: Generated text for the report section.
        """
        generation_config = GenerationConfig(
            **{
                **self.generation_args,
                "bos_token_id": model.config.bos_token_id,
                "eos_token_id": model.config.eos_token_id,
                "pad_token_id": model.config.pad_token_id,
                "decoder_start_token_id": tokenizer.cls_token_id
            }
        )
        
        generated_ids = model.generate(
            pixel_values,
            generation_config=generation_config
        )
        
        return tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

    def _run(
        self,
        image_path: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[str, Dict]:
        """Generate a comprehensive chest X-ray report containing both findings and impression.

        Args:
            image_path (str): The path to the chest X-ray image file.
            run_manager (Optional[CallbackManagerForToolRun]): The callback manager.

        Returns:
            Tuple[str, Dict]: A tuple containing the complete report and metadata.
        """
        try:
            # Process image for both models
            findings_pixels = self._process_image(image_path, self.findings_processor, self.findings_model)
            impression_pixels = self._process_image(image_path, self.impression_processor, self.impression_model)
            
            # Generate both sections
            with torch.no_grad():
                findings_text = self._generate_report_section(
                    findings_pixels, self.findings_model, self.findings_tokenizer
                )
                impression_text = self._generate_report_section(
                    impression_pixels, self.impression_model, self.impression_tokenizer
                )
            
            # Combine into formatted report
            report = (
                "CHEST X-RAY REPORT\n\n"
                f"FINDINGS:\n{findings_text}\n\n"
                f"IMPRESSION:\n{impression_text}"
            )
            
            metadata = {
                "image_path": image_path,
                "analysis_status": "completed",
                "sections_generated": ["findings", "impression"],
            }
            
            return report, metadata
            
        except Exception as e:
            return f"Error generating report: {str(e)}", {
                "image_path": image_path,
                "analysis_status": "failed",
                "error": str(e)
            }

    async def _arun(
        self,
        image_path: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Tuple[str, Dict]:
        """Asynchronously generate a comprehensive chest X-ray report."""
        return self._run(image_path)
