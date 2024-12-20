<h1 align="center">
🤖 MedMAX: Medical Multimodal Agent for X-ray 🏥
</h1>
<br>

## Problem 
Medical professionals face significant challenges when using traditional Large Language Models (LLMs) for X-ray analysis. Standard LLMs often hallucinate, lack specialized medical imaging capabilities, and can miss critical diagnostic details. While separate tools exist for various aspects of X-ray analysis, the current fragmented approach requires doctors to juggle multiple systems, leading to inefficient workflows and potential oversights in patient care.
<br>
<br>

## Our Solution
MedMAX is an intelligent medical assistant that seamlessly integrates an LLM with specialized X-ray analysis tools, providing a unified interface for comprehensive X-ray analysis. Through natural conversation, medical professionals can leverage powerful tools while the system intelligently coordinates their usage behind the scenes.

Our comprehensive toolset includes:
- **ChestXRayReportGenerator**: Generates detailed, accurate medical reports from X-ray images
- **ChestXRayClassifier**: Analyzes images for 18 different pathologies including Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, and more, providing probability scores for each condition
- **ChestXRaySegmentation**: Precisely segments anatomical structures including Left/Right Clavicle, Lungs, Heart, Aorta, and Spine
- **XRayPhraseGrounding**: Locates and visualizes specific medical findings in X-rays with bounding box precision
- **ImageVisualizer**: Enhances and displays X-ray images for optimal viewing
- **ChestXRayGenerator**: Generates synthetic chest X-rays for educational purposes
- **DicomProcessor**: Handles DICOM file processing and analysis
<br>

## Technical Implementation
MedMAX is built on a robust technical foundation:
- **Core Architecture**: Leverages LangChain and LangGraph for sophisticated agent orchestration
- **Language Model**: Powered by OpenAI's API for natural language understanding and generation
- **Specialized Tools**: Integrates medical-domain fine-tuned models for various analysis tasks
- **Interface**: Built with Gradio for an intuitive, chat-based user experience
- **Modular Design**: Allows easy integration of additional specialized medical tools
<br>

## Potential Impact
- Accelerates X-ray analysis while maintaining high accuracy
- Reduces the likelihood of missed diagnoses through multi-tool verification
- Provides valuable educational support for medical students and residents
- Offers a scalable solution for facilities with limited specialist availability
- Improves patient outcomes through comprehensive analysis
- Streamlines workflow for medical professionals
<br>

## Setup and Usage

### Prerequisites
- GPU required for optimal performance
- Python 3.8+
- OpenAI API key

### Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/medmax.git
cd medmax
```

2. Install dependencies:
```bash
pip install -e .
```

3. Set up environment variables:
```bash
echo "OPENAI_API_KEY=your_key_here" > .env
```

### Running the Application
1. Start the application:
```bash
python main.py
```

2. Open the provided Gradio interface link in your browser
3. Upload an X-ray image (supports both DICOM and standard formats)
4. Interact naturally with the agent through the chat interface
<br>

## Contributors
- Adibvafa Fallahpour
- Jun Ma
- Hongwei Lyu
<br>

---
<p align="center">
Made with ❤️ in Toronto
</p>
