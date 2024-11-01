from typing import Dict, List, Optional, Tuple, Type, Any
from pathlib import Path
import uuid
import tempfile

import numpy as np
import torch
import torchvision
import torchxrayvision as xrv
import matplotlib.pyplot as plt
import skimage.io
import skimage.measure
import skimage.transform

from pydantic import BaseModel, Field
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool


class ChestXRaySegmentationInput(BaseModel):
    """Input schema for the Chest X-Ray Segmentation Tool."""

    image_path: str = Field(..., description="Path to the chest X-ray image file to be segmented")


class OrganMetrics(BaseModel):
    """Detailed metrics for a segmented organ."""

    # Basic metrics
    area_pixels: int = Field(..., description="Area in pixels")
    area_cm2: float = Field(..., description="Approximate area in cm²")
    centroid: Tuple[float, float] = Field(..., description="(y, x) coordinates of centroid")
    bbox: Tuple[int, int, int, int] = Field(
        ..., description="Bounding box coordinates (min_y, min_x, max_y, max_x)"
    )

    # Size metrics
    width: int = Field(..., description="Width of the organ in pixels")
    height: int = Field(..., description="Height of the organ in pixels")
    aspect_ratio: float = Field(..., description="Height/width ratio")

    # Position metrics
    relative_position: Dict[str, float] = Field(
        ..., description="Position relative to image boundaries (0-1 scale)"
    )

    # Analysis metrics
    mean_intensity: float = Field(..., description="Mean pixel intensity in the organ region")
    std_intensity: float = Field(..., description="Standard deviation of pixel intensity")
    confidence_score: float = Field(..., description="Model confidence score for this organ")


class ChestXRaySegmentationTool(BaseTool):
    """Tool for performing detailed segmentation analysis of chest X-ray images.

    This tool segments a chest X-ray image into its anatomical components and provides:
    1. A visualization of all segmented organs saved as an image file
    2. Comprehensive quantitative analysis for each detected organ

    The tool analyzes 14 anatomical structures including lungs, heart, spine, and major vessels.
    For each structure, it provides metrics on size, position, and intensity characteristics.

    The output includes:
    - Multi-panel visualization saved as image file
    - Detailed metrics for each detected organ
    - Processing metadata and analysis status

    The tool integrates with torchxrayvision's PSPNet model for accurate medical image segmentation.
    """

    name: str = "chest_xray_segmentation"
    description: str = (
        "Analyzes chest X-ray images to identify and measure 14 anatomical structures: "
        "Left/Right Clavicle (collar bones), Left/Right Scapula (shoulder blades), Left/Right Lung, "
        "Left/Right Hilus Pulmonis (lung roots), Heart, Aorta, Facies Diaphragmatica (diaphragm), "
        "Mediastinum (central cavity), Weasand (esophagus), and Spine. "
        "Generates visualization showing original X-ray and segmentation masks. "
        "Returns segmentation image path and comprehensive metrics for each organ including size (area, dimensions), "
        "position (centroid, bounding box), intensity statistics, and model confidence scores. "
        "Input: Path to chest X-ray image. "
        "Output: Dict with segmentation visualization path, per-organ metrics, and analysis metadata."
    )
    args_schema: Type[BaseModel] = ChestXRaySegmentationInput

    model: Any = None
    transform: Any = None
    pixel_spacing_mm: float = 0.7
    temp_dir: Path = Path("tmp")

    def __init__(self):
        """Initialize the segmentation tool with model and temporary directory."""
        super().__init__()
        self.model = xrv.baseline_models.chestx_det.PSPNet()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()

        self.transform = torchvision.transforms.Compose(
            [xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(512)]
        )

        self.temp_dir = Path("tmp") if Path("tmp").exists() else Path(tempfile.mkdtemp())
        if not self.temp_dir.exists():
            print(f"Initialized segmentation tool with temp directory: {self.temp_dir}")

    def _compute_organ_metrics(
        self, mask: np.ndarray, original_img: np.ndarray, confidence: float
    ) -> Optional[OrganMetrics]:
        """Compute comprehensive metrics for a single organ mask."""
        # Resize mask to match original image if needed
        if mask.shape != original_img.shape:
            mask = skimage.transform.resize(
                mask, original_img.shape, order=0, preserve_range=True, anti_aliasing=False
            )

        props = skimage.measure.regionprops(mask.astype(int))
        if not props:
            return None

        props = props[0]
        area_cm2 = mask.sum() * (self.pixel_spacing_mm / 10) ** 2

        img_height, img_width = mask.shape
        cy, cx = props.centroid
        relative_pos = {
            "top": cy / img_height,
            "left": cx / img_width,
            "center_dist": np.sqrt(((cy / img_height - 0.5) ** 2 + (cx / img_width - 0.5) ** 2)),
        }

        organ_pixels = original_img[mask > 0]
        mean_intensity = organ_pixels.mean() if len(organ_pixels) > 0 else 0
        std_intensity = organ_pixels.std() if len(organ_pixels) > 0 else 0

        return OrganMetrics(
            area_pixels=int(mask.sum()),
            area_cm2=float(area_cm2),
            centroid=(float(cy), float(cx)),
            bbox=tuple(map(int, props.bbox)),
            width=int(props.bbox[3] - props.bbox[1]),
            height=int(props.bbox[2] - props.bbox[0]),
            aspect_ratio=float(
                (props.bbox[2] - props.bbox[0]) / max(1, props.bbox[3] - props.bbox[1])
            ),
            relative_position=relative_pos,
            mean_intensity=float(mean_intensity),
            std_intensity=float(std_intensity),
            confidence_score=float(confidence),
        )

    def _save_visualization(
        self, original_img: np.ndarray, pred_masks: torch.Tensor, organ_names: List[str]
    ) -> str:
        """Save visualization of original image and all segmentation masks."""
        n_organs = len(organ_names)
        n_cols = 5
        n_rows = (n_organs + 2) // n_cols

        plt.figure(figsize=(20, 5 * n_rows))

        # Plot original image
        plt.subplot(n_rows, n_cols, 1)
        plt.imshow(original_img, cmap="gray")
        plt.title("Original Image")
        plt.axis("off")

        # Plot each organ mask
        for i, organ_name in enumerate(organ_names):
            plt.subplot(n_rows, n_cols, i + 2)
            plt.imshow(pred_masks[0, i].cpu().numpy())
            plt.title(organ_name)
            plt.axis("off")

        plt.tight_layout(pad=2)

        # Save figure
        save_path = self.temp_dir / f"segmentation_{uuid.uuid4().hex[:8]}.png"
        plt.savefig(save_path)
        plt.close()

        return str(save_path)

    def _run(
        self,
        image_path: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """Run segmentation analysis and save visualization."""
        try:
            # Load and process image
            original_img = skimage.io.imread(image_path)
            if len(original_img.shape) > 2:
                original_img = original_img[:, :, 0]

            img = xrv.datasets.normalize(original_img, 255)
            img = img[None, ...]
            img = self.transform(img)
            img = torch.from_numpy(img)

            if torch.cuda.is_available():
                img = img.cuda()

            # Generate predictions
            with torch.no_grad():
                pred = self.model(img)

            # Apply sigmoid and thresholding
            pred_probs = torch.sigmoid(pred)
            pred_masks = (pred_probs > 0.5).float()

            # Save visualization
            viz_path = self._save_visualization(original_img, pred_masks, self.model.targets)

            # Compute metrics
            results = {}
            for idx, organ_name in enumerate(self.model.targets):
                mask = pred_masks[0, idx].cpu().numpy()
                if mask.sum() > 0:
                    metrics = self._compute_organ_metrics(
                        mask, original_img, float(pred_probs[0, idx].mean().cpu())
                    )
                    if metrics:
                        results[organ_name] = metrics

            metadata = {
                "image_path": image_path,
                "segmentation_image_path": viz_path,
                "original_size": original_img.shape,
                "model_size": tuple(img.shape[-2:]),
                "pixel_spacing_mm": self.pixel_spacing_mm,
                "processed_organs": list(results.keys()),
                "analysis_status": "completed",
            }

            return {
                "segmentation_image_path": viz_path,
                "metrics": {organ: metrics.dict() for organ, metrics in results.items()},
                "metadata": metadata,
            }

        except Exception as e:
            import traceback

            return {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "metadata": {"image_path": image_path, "analysis_status": "failed"},
            }

    async def _arun(
        self,
        image_path: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """Async version of _run."""
        return self._run(image_path)
