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

    async def process_message(self, 
                        message: str, 
                        display_image: Optional[str], 
                        chat_history: List[ChatMessage]) -> List[ChatMessage]:
        chat_history = chat_history or []
        
        # Use original file path if available, otherwise use display image path
        image_path = self.original_file_path or display_image
        if image_path:
            message = f"{message} `{image_path}`"
        
        # Use original file path for LLM prompt
        if self.original_file_path:
            message = f"{message} `{self.original_file_path}`"
        
        # Initialize thread if needed
        if not self.current_thread_id:
            self.current_thread_id = str(time.time())
        
        chat_history.append(ChatMessage(role="user", content=message))
        yield chat_history, self.display_file_path
        
        try:
            for event in self.agent.workflow.stream(
                {"messages": [{"role": "user", "content": message}]},
                {"configurable": {"thread_id": self.current_thread_id}}
            ):
                if isinstance(event, dict):
                    if 'process' in event:
                        content = event['process']['messages'][-1].content
                        if content:
                            chat_history.append(ChatMessage(
                                role="assistant",
                                content=content
                            ))
                            yield chat_history, self.display_file_path

                    elif 'execute' in event:
                        for message in event['execute']['messages']:
                            tool_name = message.name
                            tool_result = eval(message.content)[0]
                            
                            # For image_visualizer, use display path
                            if tool_name == "image_visualizer":
                                self.display_file_path = tool_result['image_path']
                            
                            if tool_result:
                                formatted_result = ' '.join(line.strip() for line in str(tool_result).splitlines()).strip()
                                chat_history.append(ChatMessage(
                                    role="assistant",
                                    content=formatted_result,
                                    metadata={"title": f"🔧 Using tool: {tool_name}"},
                                ))
                                yield chat_history, self.display_file_path
                            
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
            interface.process_message,
            inputs=[txt, image_display, chatbot],
            outputs=[chatbot, image_display]
        )
        bot_msg = chat_msg.then(
            lambda: gr.Textbox(interactive=True),
            None,
            [txt]
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