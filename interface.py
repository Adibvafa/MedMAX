import re
import gradio as gr
from pathlib import Path
import time
import shutil
from typing import AsyncGenerator, List, Optional, Tuple
from gradio import ChatMessage

class ChatInterface:
    def __init__(self, agent, tools_dict):
        self.agent = agent
        self.tools_dict = tools_dict
        self.upload_dir = Path("temp")
        self.upload_dir.mkdir(exist_ok=True)
        self.current_thread_id = None
        # Separate storage for original and display paths
        self.original_file_path = None  # For LLM (.dcm or other)
        self.display_file_path = None   # For UI (always viewable format)

    def handle_upload(self, file_path: str) -> str:
        """
        Handle new file upload and set appropriate paths
        Returns: display_path for UI
        """
        if not file_path:
            return None
            
        source = Path(file_path)
        timestamp = int(time.time())
        
        # Save original file with proper suffix
        suffix = source.suffix.lower()
        saved_path = self.upload_dir / f"upload_{timestamp}{suffix}"
        shutil.copy2(file_path, saved_path)  # Use file_path directly instead of source
        self.original_file_path = str(saved_path)
        
        # Handle DICOM conversion for display only
        if suffix == '.dcm':
            output, _ = self.tools_dict["DicomProcessorTool"]._run(str(saved_path))
            self.display_file_path = output['image_path']
        else:
            self.display_file_path = str(saved_path)
        
        return self.display_file_path

    def add_message(self, message, display_image, history):
        image_path = self.original_file_path or display_image
        if image_path is not None:
            history.append({"role": "user", "content": {"path": image_path}})
        if message is not None:
            history.append({"role": "user", "content": message})
        return history, gr.Textbox(value=message, interactive=False)

    async def process_message(self, 
                        message: str, 
                        display_image: Optional[str], 
                        chat_history: List[ChatMessage]):
        chat_history = chat_history or []
        
      
        # Initialize thread if needed
        if not self.current_thread_id:
            self.current_thread_id = str(time.time())

        messages = []
        image_path = self.original_file_path or display_image
        if image_path is not None:
            messages.append({"role": "user", "content": f"path: {image_path}"})
        if message is not None:
            messages.append({"role": "user", "content": message})

        try:
            for event in self.agent.workflow.stream(
                {"messages": messages},
                {"configurable": {"thread_id": self.current_thread_id}}
            ):
                if isinstance(event, dict):
                    if 'process' in event:
                        content = event['process']['messages'][-1].content
                        if content:
                            content = re.sub(r'temp/[^\s]*', '', content)
                            chat_history.append(ChatMessage(
                                role="assistant",
                                content=content
                            ))
                            yield chat_history, self.display_file_path, ""

                    elif 'execute' in event:
                        for message in event['execute']['messages']:
                            tool_name = message.name
                            tool_result = eval(message.content)[0]
                            
                            # For image_visualizer, use display path
                            if tool_name == "image_visualizer":
                                self.display_file_path = tool_result['image_path']
                                metadata={
                                    "title": f"🖼️ Image from tool: {tool_name}"
                                }

                                if tool_result:
                                    formatted_result = ' '.join(line.strip() for line in str(tool_result).splitlines()).strip()
                                    metadata["description"] = formatted_result
                                    chat_history.append(ChatMessage(
                                        role="assistant",
                                        content=formatted_result,
                                        metadata=metadata,
                                    ))
                                chat_history.append(ChatMessage(
                                    role="assistant",
                                    # content=gr.Image(value=self.display_file_path),  
                                    content = {"path": self.display_file_path},
                                    )
                                )

                                yield  chat_history, self.display_file_path, ""
                            
        except Exception as e:
            chat_history.append(ChatMessage(
                role="assistant",
                content=f"❌ Error: {str(e)}",
                metadata={"title": "Error"}
            ))
            yield chat_history, self.display_file_path

def create_demo(agent, tools_dict):
    interface = ChatInterface(agent, tools_dict)
    
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        with gr.Column():
            gr.Markdown("""
            # 🏥 MedMAX
            Multimodal Medical Agent for Chest X-rays
            """)
            # gr.Markdown(
            #     '''
            #     <div style="text-align:center; margin-bottom:20px;">
            #         <span style="font-size:3em; font-weight:bold;">Multimodal Medical Agent for Chest X-rays 🔥</span>
            #     </div>
            #     <div style="text-align:center; margin-bottom:10px;">
            #         <span style="font-size:1.5em; font-weight:bold;">MedSAM2-Segment Anything in Medical Images and Videos: Benchmark and Deployment</span>
            #     </div>
            #     <div style="text-align:center; margin-bottom:20px;">
            #         <a href="https://github.com/bowang-lab/MedSAM/tree/MedSAM2">
            #             <img src="https://badges.aleen42.com/src/github.svg" alt="GitHub" style="display:inline-block; margin-right:10px;">
            #         </a>
            #         <a href="https://arxiv.org/abs/2408.03322">
            #             <img src="https://img.shields.io/badge/arXiv-2408.03322-green?style=plastic" alt="Paper" style="display:inline-block; margin-right:10px;">
            #         </a>
            #         <a href="https://github.com/bowang-lab/MedSAMSlicer/tree/SAM2">
            #             <img src="https://img.shields.io/badge/3D-Slicer-Plugin" alt="3D Slicer Plugin" style="display:inline-block; margin-right:10px;">
            #         </a>
            #         <a href="https://drive.google.com/drive/folders/1EXzRkxZmrXbahCFA8_ImFRM6wQDEpOSe?usp=sharing">
            #             <img src="https://img.shields.io/badge/Video-Tutorial-green?style=plastic" alt="Video Tutorial" style="display:inline-block; margin-right:10px;">
            #         </a>
            #         <a href="https://github.com/bowang-lab/MedSAM/tree/MedSAM2?tab=readme-ov-file#fine-tune-sam2-on-the-abdomen-ct-dataset">
            #             <img src="https://img.shields.io/badge/Fine--tune-SAM2-blue" alt="Fine-tune SAM2" style="display:inline-block; margin-right:10px;">
            #         </a>
            #     </div>
            #     <div style="text-align:left; margin-bottom:20px;">
            #         This API supports using box (generated by scribble) and point prompts for video segmentation with 
            #         <a href="https://ai.meta.com/sam2/" target="_blank">SAM2</a>. Welcome to join our <a href="https://forms.gle/hk4Efp6uWnhjUHFP6" target="_blank">mailing list</a> to get updates or send feedback.
            #     </div>
            #     <div style="margin-bottom:20px;">
            #         <ol style="list-style:none; padding-left:0;">
            #             <li>1. Upload video file</li>
            #             <li>2. Select model size and downsample frame rate and run <b>Preprocess</b></li>
            #             <li>3. Use <b>Stroke to Box Prompt</b> to draw box on the first frame or <b>Point Prompt</b> to click on the first frame.</li>
            #             <li>&nbsp;&nbsp;&nbsp;Note: The bounding rectangle of the stroke should be able to cover the segmentation target.</li>
            #             <li>4. Click <b>Segment</b> to get the segmentation result</li>
            #             <li>5. Click <b>Add New Object</b> to add new object</li>
            #             <li>6. Click <b>Start Tracking</b> to track objects in the video</li>
            #             <li>7. Click <b>Reset</b> to reset the app</li>
            #             <li>8. Download the video with segmentation results</li>
            #         </ol>
            #     </div>
            #     <div style="text-align:left; line-height:1.8;">
            #         We designed this API and <a href="https://github.com/bowang-lab/MedSAMSlicer/tree/SAM2" target="_blank">3D Slicer Plugin</a> for medical image and video segmentation where the checkpoints are based on the original SAM2 models (<a href="https://github.com/facebookresearch/segment-anything-2" target="_blank">https://github.com/facebookresearch/segment-anything-2</a>). The image segmentation fine-tune code has been released on <a href="https://github.com/bowang-lab/MedSAM/tree/MedSAM2?tab=readme-ov-file#fine-tune-sam2-on-the-abdomen-ct-dataset" target="_blank">GitHub</a>. The video fine-tuning code is under active development and will be released as well.  
            #     </div>
            #     <div style="text-align:left; line-height:1.8;">
            #         If you find these tools useful, please consider citing the following papers:
            #     </div>
            #     <div style="text-align:left; line-height:1.8;">
            #         Ravi, N., Gabeur, V., Hu, Y.T., Hu, R., Ryali, C., Ma, T., Khedr, H., Rädle, R., Rolland, C., Gustafson, L., Mintun, E., Pan, J., Alwala, K.V., Carion, N., Wu, C.Y., Girshick, R., Dollár, P., Feichtenhofer, C.: SAM 2: Segment Anything in Images and Videos. arXiv:2408.00714 (2024)
            #     </div>            
            #     <div style="text-align:left; line-height:1.8;">
            #         Ma, J., Kim, S., Li, F., Baharoon, M., Asakereh, R., Lyu, H., Wang, B.: Segment Anything in Medical Images and Videos: Benchmark and Deployment. arXiv preprint arXiv:2408.03322 (2024)
            #     </div> 
            #     <div style="text-align:left; line-height:1.8;"> 
            #         Other useful resources: 
            #         <a href="https://ai.meta.com/sam2" target="_blank">Official demo</a> from MetaAI, 
            #         <a href="https://www.youtube.com/watch?v=Dv003fTyO-Y" target="_blank">Video tutorial</a> from Piotr Skalski.
            #     </div>
            #     '''
            # )
            
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        [],
                        height=800,
                        container=True,
                        show_label=True,
                        elem_classes="chat-box",
                        type="messages",
                        label="Agent",
                        avatar_images=(None, "https://em-content.zobj.net/source/twitter/53/robot-face_1f916.png")
                    )
                    with gr.Row():
                        with gr.Column(scale=3):
                            txt = gr.Textbox(
                                show_label=False,
                                placeholder="Ask about the X-ray...",
                                container=False
                            )
                            
                with gr.Column(scale=3):
                    image_display = gr.Image(
                        label="Image",
                        type="filepath",
                        height=700,
                        container=True
                    )
                    with gr.Row():
                        upload_button = gr.UploadButton(
                            "📎 Upload X-Ray",
                            file_types=["image"],
                        )
                        dicom_upload = gr.UploadButton(
                            "📄 Upload DICOM",
                            file_types=["file"],
                        )
                    with gr.Row():
                        clear_btn = gr.Button("Clear Chat")
                        new_thread_btn = gr.Button("New Thread")

        # Event handlers
        def clear_chat():
            interface.original_file_path = None
            interface.display_file_path = None
            return [], None

        def new_thread():
            interface.current_thread_id = str(time.time())
            return [], interface.display_file_path

        def handle_file_upload(file):
            return interface.handle_upload(file.name)

        chat_msg = txt.submit(
            interface.add_message,
            inputs=[txt, image_display, chatbot],
            outputs=[chatbot, txt]
        )
        bot_msg = chat_msg.then(
            interface.process_message,
            inputs=[txt, image_display, chatbot],
            outputs=[chatbot, image_display, txt]
        )


        upload_button.upload(
            handle_file_upload,
            inputs=upload_button,
            outputs=image_display
        )
        
        dicom_upload.upload(
            handle_file_upload,
            inputs=dicom_upload,
            outputs=image_display
        )

        clear_btn.click(clear_chat, outputs=[chatbot, image_display])
        new_thread_btn.click(new_thread, outputs=[chatbot, image_display])

    return demo