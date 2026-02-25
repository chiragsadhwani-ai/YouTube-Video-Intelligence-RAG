import gradio as gr
from ingestion_service import get_raw_transcript, format_transcript_content
from langchain_orchestrator import LangChainAssistant

# Initialize the LangChain Orchestrator
assistant = LangChainAssistant()

def process_request(url, question):
    if not url.strip() or not question.strip():
        return "⚠️ Please provide both a YouTube URL and a question."
    
    raw_data = get_raw_transcript(url)
    if not raw_data:
        return "❌ System Error: Could not retrieve YouTube transcript. Please check the URL and try again."
    
    document_text = format_transcript_content(raw_data)
    vector_db = assistant.create_vector_store(document_text)
    response = assistant.run_qa_chain(question, vector_db)
    
    return response.content

CSS = """
    .gradio-container {
        max-width: 98vw !important;
        width: 98vw !important;
        margin: 0 auto !important;
        font-family: 'Inter', sans-serif !important;
    }
    .header-box {
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 20px;
        margin-bottom: 8px;
    }
    .input-panel {
        background: #f8fafc;
        border-radius: 10px;
        padding: 24px;
        border: 1px solid #e2e8f0;
    }
    .output-panel {
        background: #ffffff;
        border-radius: 10px;
        padding: 24px;
        border: 1px solid #e2e8f0;
    }
    footer { display: none !important; }

    /* Force the badge to top-right of the actual viewport */
    #user-badge {
        position: fixed !important;
        top: 16px !important;
        right: 24px !important;
        left: auto !important;
        text-align: right !important;
        z-index: 99999 !important;
        background: white !important;
        padding: 8px 14px !important;
        border-radius: 8px !important;
        border: 1px solid #e2e8f0 !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08) !important;
        line-height: 1.6 !important;
        width: fit-content !important;
    }
    #user-badge .name {
        font-weight: 600;
        font-size: 14px;
        color: #1e293b;
        display: block;
    }
    #user-badge .email {
        font-size: 12px;
        color: #64748b;
        display: block;
    }    
"""

with gr.Blocks(title="Video Intelligence Platform", css=CSS) as ui:

    gr.HTML("""
        <div id="user-badge">   <!-- ← was class="user-info" -->
            <span class="name">Chirag Sadhwani</span>
            <span class="email">chiragsadhwani78@email.com</span>
        </div>
    """)


    with gr.Column(elem_classes="header-box"):
        gr.Markdown(
            """
            # Video Intelligence Platform
            **Powered by LangChain** — Extract insights from any YouTube video using AI-driven Q&A.
            """
        )

    gr.Markdown("<br>")

    with gr.Row(equal_height=True):

        # LEFT: Input panel
        with gr.Column(elem_classes="input-panel", scale=1):
            gr.Markdown("### 📥 Input")
            youtube_url = gr.Textbox(
                label="YouTube URL",
                placeholder="https://www.youtube.com/watch?v=...",
            )
            gr.Markdown("<br>")
            user_query = gr.Textbox(
                label="Your Question",
                placeholder="What topics are covered in this video?",
                lines=3,
            )
            gr.Markdown("<br>")
            submit_btn = gr.Button("Run Analysis →", variant="primary")

        # RIGHT: Output panel
        with gr.Column(elem_classes="output-panel", scale=1):
            gr.Markdown("### 📤 AI Response")
            output = gr.Textbox(
                label="",
                lines=10,
                placeholder="Your answer will appear here after analysis...",
                show_label=False,
            )

    gr.Markdown(
        "<p style='text-align:center; color:#94a3b8; font-size:13px; margin-top:24px;'>"
        "Results are generated from video transcript data. Accuracy depends on transcript availability."
        "</p>"
    )

    submit_btn.click(
        fn=process_request,
        inputs=[youtube_url, user_query],
        outputs=output,
    )

if __name__ == "__main__":
    ui.launch(theme=gr.themes.Base())
