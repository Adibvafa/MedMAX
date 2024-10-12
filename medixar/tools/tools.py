from typing import Dict, Optional, Tuple, Type
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


class ChestXRayInput(BaseModel):
    """Input for the Chest X-Ray Classifier tool."""

    image_path: str = Field(..., description="Path to the chest X-ray image file")


class ChestXRayClassifierTool(BaseTool):
    """Tool that classifies chest X-ray images for multiple pathologies.

    This tool uses a pre-trained DenseNet model to analyze chest X-ray images and
    predict the likelihood of various pathologies. The model can classify the following 18 conditions:

    Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema,
    Enlarged Cardiomediastinum, Fibrosis, Fracture, Hernia, Infiltration,
    Lung Lesion, Lung Opacity, Mass, Nodule, Pleural Thickening, Pneumonia, Pneumothorax

    The output values represent the probability (from 0 to 1) of each condition being present in the image.
    A higher value indicates a higher likelihood of the condition being present.

    Setup:
        Ensure you have the necessary dependencies installed:

        .. code-block:: bash

            pip install -U langchain-core pydantic torch torchvision torchxrayvision scikit-image

    Instantiate:

        .. code-block:: python

            from your_module import ChestXRayClassifierTool

            tool = ChestXRayClassifierTool()

    Invoke directly with args:

        .. code-block:: python

            tool.invoke({'image_path': '/path/to/chest_xray.jpg'})

        .. code-block:: python

            {'Atelectasis': 0.5211657, 'Cardiomegaly': 0.18291874, ...}  # Example output

    Note: The output values are probabilities between 0 and 1. A value closer to 1 indicates
    a higher likelihood of the condition being present in the X-ray image.
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


class RadiologyImageInput(BaseModel):
    """Input for the Radiology Report Generator tool."""

    image_path: str = Field(..., description="Path to the radiology image file")


class RadiologyReportGeneratorTool(BaseTool):
    """Tool that generates a radiology report based on an input image.

    This tool is designed to analyze radiology images and produce corresponding reports.
    Currently, it returns a placeholder string, but it's structured to be extended
    with actual image analysis functionality in the future.

    Setup:
        Ensure you have the necessary dependencies installed:

        .. code-block:: bash

            pip install -U langchain-core pydantic

    Instantiate:

        .. code-block:: python

            from your_module import RadiologyReportGeneratorTool

            tool = RadiologyReportGeneratorTool()

    Invoke directly with args:

        .. code-block:: python

            tool.invoke({'image_path': '/path/to/image.jpg'})

        .. code-block:: python

            'Radiology report for image: /path/to/image.jpg, heart is normal, lungs have pneumonia.'  # Placeholder output

    """

    name: str = "radiology_report_generator"
    description: str = (
        "A tool that analyzes radiology images and generates corresponding reports. "
        "Input should be the path to a radiology image file."
    )
    args_schema: Type[BaseModel] = RadiologyImageInput

    def _run(
        self,
        image_path: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[str, Dict]:
        """Generate a radiology report based on the input image.

        Args:
            image_path (str): The path to the radiology image file.
            run_manager (Optional[CallbackManagerForToolRun]): The callback manager for the tool run.

        Returns:
            Tuple[str, Dict]: A tuple containing the generated report (currently a placeholder)
                              and any additional metadata.

        Raises:
            Exception: If there's an error processing the image.
        """
        try:
            # Placeholder for future image analysis logic
            report = f"Radiology report for image: {image_path}, heart is normal, lungs have pneumonia."
            metadata = {"image_path": image_path, "analysis_status": "placeholder"}
            return report, metadata
        except Exception as e:
            return f"Error generating report: {str(e)}", {}

    async def _arun(
        self,
        image_path: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Tuple[str, Dict]:
        """Asynchronously generate a radiology report based on the input image.

        Args:
            image_path (str): The path to the radiology image file.
            run_manager (Optional[AsyncCallbackManagerForToolRun]): The async callback manager for the tool run.

        Returns:
            Tuple[str, Dict]: A tuple containing the generated report (currently a placeholder)
                              and any additional metadata.

        Raises:
            Exception: If there's an error processing the image.
        """
        try:
            # Placeholder for future async image analysis logic
            report = f"Radiology report for image: {image_path}, heart is normal, lungs have pneumonia."
            metadata = {"image_path": image_path, "analysis_status": "placeholder"}
            return report, metadata
        except Exception as e:
            return f"Error generating report: {str(e)}", {}


class OrganSizeMeasurementInput(BaseModel):
    """Input for the Organ Size Measurement tool."""

    image_path: str = Field(..., description="Path to the radiology image file")
    organ: str = Field(..., description="Name of the organ of interest")


class OrganSizeMeasurementTool(BaseTool):
    """Tool that measures the size of a specified organ in a radiology image.

    This tool is designed to analyze radiology images and return the size of a specified organ.
    Currently, it returns a placeholder size, but it's structured to be extended
    with actual image analysis functionality in the future.

    Setup:
        Ensure you have the necessary dependencies installed:

        .. code-block:: bash

            pip install -U langchain-core pydantic

    Instantiate:

        .. code-block:: python

            from your_module import OrganSizeMeasurementTool

            tool = OrganSizeMeasurementTool()

    Invoke directly with args:

        .. code-block:: python

            tool.invoke({'image_path': '/path/to/image.jpg', 'organ': 'liver'})

        .. code-block:: python

            'The size of the liver in the image /path/to/image.jpg is approximately 15 cm x 10 cm x 8 cm.'  # Placeholder output

    """

    name: str = "organ_size_measurement"
    description: str = (
        "A tool that analyzes radiology images and measures the size of a specified organ. "
        "Input should be the path to a radiology image file and the name of the organ of interest."
    )
    args_schema: Type[BaseModel] = OrganSizeMeasurementInput

    def _run(
        self,
        image_path: str,
        organ: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[str, Dict]:
        """Measure the size of a specified organ in the input image.

        Args:
            image_path (str): The path to the radiology image file.
            organ (str): The name of the organ to measure.
            run_manager (Optional[CallbackManagerForToolRun]): The callback manager for the tool run.

        Returns:
            Tuple[str, Dict]: A tuple containing the size measurement (currently a placeholder)
                              and any additional metadata.

        Raises:
            Exception: If there's an error processing the image or measuring the organ.
        """
        try:
            # Placeholder for future organ measurement logic
            size = f"15 cm x 10 cm x 8 cm"
            result = f"The size of the {organ} in the image {image_path} is approximately {size}."
            metadata = {
                "image_path": image_path,
                "organ": organ,
                "measurement_status": "placeholder",
            }
            return result, metadata
        except Exception as e:
            return f"Error measuring organ size: {str(e)}", {}

    async def _arun(
        self,
        image_path: str,
        organ: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Tuple[str, Dict]:
        """Asynchronously measure the size of a specified organ in the input image.

        Args:
            image_path (str): The path to the radiology image file.
            organ (str): The name of the organ to measure.
            run_manager (Optional[AsyncCallbackManagerForToolRun]): The async callback manager for the tool run.

        Returns:
            Tuple[str, Dict]: A tuple containing the size measurement (currently a placeholder)
                              and any additional metadata.

        Raises:
            Exception: If there's an error processing the image or measuring the organ.
        """
        try:
            # Placeholder for future async organ measurement logic
            size = f"15 cm x 10 cm x 8 cm"
            result = f"The size of the {organ} in the image {image_path} is approximately {size}."
            metadata = {
                "image_path": image_path,
                "organ": organ,
                "measurement_status": "placeholder",
            }
            return result, metadata
        except Exception as e:
            return f"Error measuring organ size: {str(e)}", {}
