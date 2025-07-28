import gradio as gr
import sys
import threading
import queue
import time
import random
from io import TextIOBase
import datetime
import subprocess
import os
import config  # Import the config module to modify its parameters

# Assuming these are in your project structure
from inference import inference_patch, postprocess_inst_names
from convert import abc2xml, xml2, pdf2img  # Make sure these functions exist in convert.py

title_html = """
<div class="title-container">
    <h1 class="title-text">NotaGen</h1>  
        <!-- ArXiv -->
        <a href="https://arxiv.org/abs/2502.18008">
            <img src="https://img.shields.io/badge/NotaGen_Paper-ArXiv-%23B31B1B?logo=arxiv&logoColor=white" alt="Paper">
        </a>
         
        <!-- GitHub -->
        <a href="https://github.com/ElectricAlexis/NotaGen">
            <img src="https://img.shields.io/badge/NotaGen_Code-GitHub-%23181717?logo=github&logoColor=white" alt="GitHub">
        </a>
         
        <!-- HuggingFace -->
        <a href="https://huggingface.co/ElectricAlexis/NotaGen">
            <img src="https://img.shields.io/badge/NotaGen_Weights-HuggingFace-%23FFD21F?logo=huggingface&logoColor=white" alt="Weights">
        </a>
         
        <!-- Web Demo -->
        <a href="https://electricalexis.github.io/notagen-demo/">
            <img src="https://img.shields.io/badge/NotaGen_Demo-Web-%23007ACC?logo=google-chrome&logoColor=white" alt="Demo">
        </a>
</div>
<p style="font-size: 1.2em;">NotaGen is a model for generating sheet music in ABC notation format. Select a period, composer, and instrumentation to generate classical-style music!</p>
"""

# Read prompt combinations
with open('prompts.txt', 'r') as f:
    prompts_data = f.readlines()  # Renamed to avoid conflict

valid_combinations = set()
for p_line in prompts_data:  # Use new variable name
    p_line = p_line.strip()
    parts = p_line.split('_')
    if len(parts) == 3:  # Basic validation
        valid_combinations.add((parts[0], parts[1], parts[2]))

# Generate available options
periods = sorted({p for p, _, _ in valid_combinations})
# Composers and instruments will be filtered dynamically

# Global stop event and thread reference
stop_event = threading.Event()
generation_thread = None


# Dynamic component updates
def update_dropdowns(period, composer):  # Renamed for clarity
    if not period:
        return [
            gr.update(choices=[], value=None, interactive=False),
            gr.update(choices=[], value=None, interactive=False)
        ]

    valid_composers_list = sorted({c for p, c, _ in valid_combinations if p == period})

    current_composer = composer
    if composer not in valid_composers_list:
        current_composer = None

    valid_instruments_list = sorted(
        {i for p, c, i in valid_combinations if p == period and c == current_composer}) if current_composer else []

    return [
        gr.update(
            choices=valid_composers_list,
            value=current_composer,
            interactive=True
        ),
        gr.update(
            choices=valid_instruments_list,
            value=None,  # Always reset instrument choice
            interactive=bool(valid_instruments_list)
        )
    ]


class RealtimeStream(TextIOBase):
    def __init__(self, q_obj):  # Renamed parameter
        self.queue = q_obj

    def write(self, text):
        self.queue.put(text)
        return len(text)

    def flush(self):
        pass


def convert_files_enhanced(raw_abc_content, period, composer, instrumentation):
    if not all([period, composer, instrumentation]):
        # This check might be redundant if generate_music already validates
        raise gr.Error("Selection criteria missing for file conversion.")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    prompt_str = f"{period}_{composer}_{instrumentation}"

    # Post-process the ABC content first
    postprocessed_abc = postprocess_inst_names(raw_abc_content)

    filename_base = f"{timestamp}_{prompt_str}_processed"  # Indicate it's processed

    processed_abc_filename = f"{filename_base}.abc"
    with open(processed_abc_filename, "w", encoding="utf-8") as f:
        f.write(postprocessed_abc)

    # Ensure output directory exists (optional, good practice)
    # os.makedirs("output_files", exist_ok=True)
    # filename_base = os.path.join("output_files", filename_base)
    # processed_abc_filename = os.path.join("output_files", processed_abc_filename)

    file_paths_dict = {'abc': processed_abc_filename}
    image_filenames_list = []

    try:
        # All conversions now use the `filename_base` (which implies `filename_base.abc` for post-processed)
        # These functions take the base name and append extensions internally
        abc2xml(filename_base)  # Creates filename_base.xml from filename_base.abc
        xml2(filename_base, 'pdf')  # Creates filename_base.pdf from filename_base.xml
        xml2(filename_base, 'mid')  # Creates filename_base.mid from filename_base.xml
        xml2(filename_base, 'mp3')  # Creates filename_base.mp3 from filename_base.xml

        images = pdf2img(filename_base)  # Creates PNGs from filename_base.pdf
        for i, image in enumerate(images):
            img_path = f"{filename_base}_page_{i + 1}.png"
            image.save(img_path, "PNG")
            image_filenames_list.append(img_path)

        file_paths_dict.update({
            'xml': f"{filename_base}.xml",
            'pdf': f"{filename_base}.pdf",
            'mid': f"{filename_base}.mid",
            'mp3': f"{filename_base}.mp3",
            'png_pages': image_filenames_list,
            'pages': len(images),
            'current_page': 0,
            'base': filename_base  # Base path for page navigation
        })

    except Exception as e:
        # Clean up partially created files if desired
        raise gr.Error(f"File processing and conversion failed: {str(e)}")
        print(e)

    return file_paths_dict, postprocessed_abc


def update_page_display(direction, current_pdf_state):
    if not current_pdf_state or 'pages' not in current_pdf_state or current_pdf_state['pages'] == 0:
        return None, gr.update(interactive=False), gr.update(interactive=False), current_pdf_state

    if direction == "prev" and current_pdf_state['current_page'] > 0:
        current_pdf_state['current_page'] -= 1
    elif direction == "next" and current_pdf_state['current_page'] < current_pdf_state['pages'] - 1:
        current_pdf_state['current_page'] += 1

    current_page_idx = current_pdf_state['current_page']

    new_image_path = None
    if current_pdf_state.get('png_pages') and 0 <= current_page_idx < len(current_pdf_state['png_pages']):
        new_image_path = current_pdf_state['png_pages'][current_page_idx]

    prev_btn_interactive = current_page_idx > 0
    next_btn_interactive = current_page_idx < current_pdf_state['pages'] - 1

    # Check if image path exists, otherwise show nothing or placeholder
    if new_image_path and not os.path.exists(new_image_path):
        print(f"Warning: Image path not found: {new_image_path}")  # Log this
        new_image_path = None  # Or a placeholder image path

    return new_image_path, gr.update(interactive=prev_btn_interactive), gr.update(
        interactive=next_btn_interactive), current_pdf_state


def generate_music(period, composer, instrumentation, num_bars, metadata_K, metadata_M, model_name_selected, seed_val,
                   top_k_val, top_p_val, temperature_val):
    global stop_event, generation_thread
    stop_event.clear()  # Clear any previous stop signals

    if generation_thread and generation_thread.is_alive():
        stop_event.set()  # Signal the old thread to stop
        generation_thread.join(timeout=5)
        if generation_thread.is_alive():  # If still alive, it's stuck, can't start new one easily
            yield "Previous generation is still stopping. Please wait and try again.", "", None, None, None, gr.update(
                value=None, visible=False)
            return
        stop_event = threading.Event()

    if not period or not composer or not instrumentation:
        raise gr.Error("Period, Composer, and Instrumentation must be selected.")
    if (period, composer, instrumentation) not in valid_combinations:
        raise gr.Error("Invalid prompt combination! Please re-select from the period options.")

    models_path = "models"
    if not os.path.exists(os.path.join(models_path, model_name_selected)) or model_name_selected == "NONE":
        raise gr.Error("Invalid Model Name! Please re-select from the config options or ensure model exists.")

    # Initialize outputs for yield
    process_log_output = ""
    final_abc_output = ""
    pdf_image_path_output = None
    audio_file_path_output = None
    pdf_state_output = None
    download_files_update = gr.update(value=None, visible=False)

    # Set random seeds
    if seed_val == -1:
        seed_val = random.randint(0, 100000000)
    random.seed(seed_val)
    try:
        import numpy as np
        np.random.seed(seed_val)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed_val)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_val)
    except ImportError:
        pass

    output_q_stream = queue.Queue()
    original_stdout = sys.stdout
    sys.stdout = RealtimeStream(output_q_stream)

    result_container = []  # To store result from thread

    def run_inference_thread():
        nonlocal result_container  # Allow modification of outer scope variable
        try:
            # Correctly pass parameters to inference_patch
            raw_abc_result = inference_patch(
                period=period, composer=composer, instrumentation=instrumentation,
                num_bars=num_bars, metadata_K=metadata_K, metadata_M=metadata_M,
                model_path=os.path.join(models_path, model_name_selected),  # Pass full path
                seed=seed_val, top_k=top_k_val, top_p=top_p_val, temperature=temperature_val,
                stop_event=stop_evt
            )
            result_container.append(raw_abc_result)
        except Exception as e:
            output_q_stream.put(f"\nError during inference: {str(e)}\n")
            result_container.append(None)  # Indicate failure
            print(e)
        finally:
            if sys.stdout == output_q_stream.queue.put.__self__:  # Check if stdout is still our stream
                sys.stdout = original_stdout

    generation_thread = threading.Thread(target=run_inference_thread, args=(stop_event,))
    generation_thread.start()

    # Stream inference process output
    while generation_thread.is_alive():
        if stop_event.is_set():
            process_log_output += "\nGeneration stop requested by user."
            break
        try:
            text = output_q_stream.get(timeout=0.1)
            process_log_output += text
            yield process_log_output, final_abc_output, pdf_image_path_output, audio_file_path_output, pdf_state_output, download_files_update
        except queue.Empty:
            continue

    generation_thread.join(timeout=2)  # Wait for thread to finish after loop

    # Process any remaining output from queue
    while not output_q_stream.empty():
        text = output_q_stream.get_nowait()
        process_log_output += text

    sys.stdout = original_stdout  # Ensure stdout is restored

    if stop_event.is_set():
        process_log_output += "\nGeneration stopped."
        yield process_log_output, "Generation stopped by user.", None, None, None, gr.update(value=None, visible=False)
        return

    raw_abc_from_inference = result_container[0] if result_container and result_container[0] is not None else ""

    if not raw_abc_from_inference:
        process_log_output += "\nInference failed or produced no ABC output."
        yield process_log_output, "Error: Inference failed or no content generated.", None, None, None, gr.update(
            value=None, visible=False)
        return

    process_log_output += "\nInference complete. Starting file conversion..."
    final_abc_output = "Converting files and generating previews..."  # Temporary message
    yield process_log_output, final_abc_output, pdf_image_path_output, audio_file_path_output, pdf_state_output, download_files_update

    try:
        converted_file_paths, final_postprocessed_abc = convert_files_enhanced(raw_abc_from_inference, period, composer,
                                                                               instrumentation)

        final_abc_output = final_postprocessed_abc
        pdf_state_output = converted_file_paths  # This holds all paths and page info

        if converted_file_paths.get('png_pages') and converted_file_paths['pages'] > 0:
            pdf_image_path_output = converted_file_paths['png_pages'][0]  # First page

        audio_file_path_output = converted_file_paths.get('mp3')

        downloadable_files_list = []
        for f_key in ['abc', 'xml', 'pdf', 'mid', 'mp3']:
            if f_key in converted_file_paths and os.path.exists(converted_file_paths[f_key]):
                downloadable_files_list.append(converted_file_paths[f_key])
        download_files_update = gr.update(value=downloadable_files_list, visible=True)
        process_log_output += "\nFile conversion successful."

    except Exception as e:
        process_log_output += f"\nError during file conversion: {str(e)}"
        final_abc_output = f"File conversion error: {str(e)}"
        print(e)
        # Reset outputs specific to files on error
        pdf_image_path_output = None
        audio_file_path_output = None
        pdf_state_output = None
        download_files_update = gr.update(value=None, visible=False)
        yield process_log_output, final_abc_output, pdf_image_path_output, audio_file_path_output, pdf_state_output, download_files_update
        return

    yield process_log_output, final_abc_output, pdf_image_path_output, audio_file_path_output, pdf_state_output, download_files_update


def stop_generation_action():
    global stop_event
    stop_event.set()


# --- Gradio UI Definition ---
with gr.Blocks() as demo:
    gr.HTML(title_html)

    pdf_state = gr.State()  # For PDF pagination

    with gr.Row():
        # Left Column: Inputs & Logs
        with gr.Column(scale=1):
            period_dd = gr.Dropdown(choices=periods, value=None, label="Period", interactive=True)
            composer_dd = gr.Dropdown(choices=[], value=None, label="Composer", interactive=False)
            instrument_dd = gr.Dropdown(choices=[], value=None, label="Instrumentation", interactive=False)

            with gr.Accordion("Config Parameters", open=False):
                model_choices = [entry.name for entry in os.scandir("models") if
                                            entry.is_file() and entry.name != "NONE"] if os.path.exists("models") else [
                    "NONE"]
                model_name_dd = gr.Dropdown(choices=model_choices, label="Model Name",
                                            value=model_choices[0] if model_choices else "NONE")
                seed_slider = gr.Slider(minimum=-1, maximum=100000000, step=1, value=-1, label="Seed",
                                        info="For Random Seed, Enter -1.")
                top_k_slider = gr.Slider(minimum=1, maximum=20, step=1, value=config.TOP_K, label="Top K")
                top_p_slider = gr.Slider(minimum=0.01, maximum=1.0, step=0.01, value=config.TOP_P, label="Top P")
                temperature_slider = gr.Slider(minimum=0.1, maximum=4.0, step=0.01, value=config.TEMPERATURE,
                                               label="Temperature")

            with gr.Accordion("Tune Parameters", open=False):
                num_bars_number = gr.Number(minimum=0, precision=0, label="Number of Bars", value=0,
                                            info="0 for model choice.")
                key_signature_dd = gr.Dropdown(
                    choices=["None", "C", "G", "D", "A", "E", "B", "F#", "C#", "F", "Bb", "Eb", "Ab", "Db", "Gb", "Cb"],
                    label="Key Signature (K:)", value="None")
                time_signature_text = gr.Textbox(label="Time Signature (M:)", value="",
                                                 placeholder="e.g., 4/4, 3/4, C, C|")

            generate_btn = gr.Button("Generate!", variant="primary")
            stop_btn = gr.Button("Stop Generating", visible=False)  # Initially hidden

            process_output_text = gr.Textbox(
                label="Generation Process Log", interactive=False, lines=10, max_lines=15,
                placeholder="Generation progress will be shown here...", elem_classes="process-output")

            final_output_abc_text = gr.Textbox(
                label="Post-processed ABC Notation", interactive=True, lines=10, max_lines=15,
                placeholder="Post-processed ABC scores will appear here...", elem_classes="final-output")

            audio_player_component = gr.Audio(
                label="Audio Preview", type="filepath", interactive=False, elem_classes="audio-panel"
            )

        # Right Column: Previews
        with gr.Column(scale=1):
            pdf_image_display = gr.Image(
                label="Sheet Music Preview", type="filepath", height=600,  # Adjust height as needed
                interactive=False, show_download_button=False, elem_id="pdf-preview"
            )
            with gr.Row(equal_height=True):
                prev_page_btn = gr.Button("⬅️ Last Page", variant="secondary", size="sm", elem_classes="page-btn",
                                          interactive=False)
                next_page_btn = gr.Button("Next Page ➡️", variant="secondary", size="sm", elem_classes="page-btn",
                                          interactive=False)

    # Bottom Section: Downloads
    with gr.Row():
        with gr.Column():
            gr.Markdown("**Download Generated Files:**")
            download_files_list = gr.Files(
                label="Download Files", visible=False, elem_classes="download-files", type="filepath"
            )

    # --- Event Handlers ---
    period_dd.change(update_dropdowns, inputs=[period_dd, composer_dd], outputs=[composer_dd, instrument_dd])
    composer_dd.change(update_dropdowns, inputs=[period_dd, composer_dd],
                       outputs=[composer_dd, instrument_dd])  # Corrected: instrument_dd should react

    generate_inputs = [
        period_dd, composer_dd, instrument_dd,
        num_bars_number, key_signature_dd, time_signature_text,
        model_name_dd, seed_slider, top_k_slider, top_p_slider, temperature_slider
    ]
    generate_outputs = [
        process_output_text, final_output_abc_text,
        pdf_image_display, audio_player_component, pdf_state,
        download_files_list
    ]

    generate_btn.click(
        lambda: (gr.update(visible=False), gr.update(visible=True)),
        inputs=[], outputs=[generate_btn, stop_btn]
    ).then(
        generate_music, inputs=generate_inputs, outputs=generate_outputs
    ).then(
        # This lambda runs after generate_music completes or errors
        lambda pdf_s: (
            gr.update(visible=True),  # generate_btn
            gr.update(visible=False),  # stop_btn
            # Update page buttons based on initial PDF state
            gr.update(interactive=(pdf_s is not None and pdf_s.get('current_page', 0) > 0)),  # prev_page_btn
            gr.update(interactive=(
                        pdf_s is not None and pdf_s.get('pages', 0) > 1 and pdf_s.get('current_page', 0) < pdf_s.get(
                    'pages', 0) - 1))  # next_page_btn
        ),
        inputs=[pdf_state],  # Pass the pdf_state to this lambda
        outputs=[generate_btn, stop_btn, prev_page_btn, next_page_btn]
    )

    stop_btn.click(
        stop_generation_action, inputs=[], outputs=[]
        # stop_generation_action could return a status to a hidden textbox if needed
    ).then(
        lambda: (gr.update(visible=False), gr.update(visible=True)),
        inputs=[], outputs=[stop_btn, generate_btn]
    )

    # PDF Page Navigation Events
    prev_page_btn.click(
        update_page_display,
        inputs=[gr.State("prev"), pdf_state],
        outputs=[pdf_image_display, prev_page_btn, next_page_btn, pdf_state]
    )
    next_page_btn.click(
        update_page_display,
        inputs=[gr.State("next"), pdf_state],
        outputs=[pdf_image_display, prev_page_btn, next_page_btn, pdf_state]
    )

# --- CSS Styling ---
css = """
.title-container { display: flex; align-items: center; gap: 15px; margin-bottom: 10px; }
.title-text { margin: 0; font-size: 1.8em; }
/* ... other styles from first demo ... */
#pdf-preview { border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
.page-btn { padding: 8px 12px !important; margin: auto !important; } /* Adjusted padding */
.page-btn:hover { background: #f0f0f0 !important; transform: scale(1.05); }
.audio-panel { margin-top: 15px !important; max-width: 100%; } /* Ensure audio player fits */
.download-files { margin-top: 15px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }

/* Styles from your current demo */
.process-output { background-color: #f0f0f0; font-family: monospace; padding: 10px; border-radius: 5px; }
.final-output { background-color: #ffffff; font-family: sans-serif; padding: 10px; border-radius: 5px; }
.process-output textarea, .final-output textarea { /* Apply to both textareas */
    max-height: 300px !important; /* Adjust as needed */
    overflow-y: auto !important;
    white-space: pre-wrap;
    word-break: break-all; /* Helps with long unbreakable strings */
}
"""
demo.css = css

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,  # Kept your port
        share=True # Uncomment if needed
        # debug=True  # Helpful during development
    )