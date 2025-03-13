import random
import gradio as gr
import sys
import threading
import queue
from io import TextIOBase
from inference import inference_patch
import datetime
import subprocess
import os
import config  # Import the config module to modify its parameters

# Predefined valid combinations set
with open('prompts.txt', 'r') as f:
    prompts = f.readlines()
valid_combinations = set()
for prompt in prompts:
    prompt = prompt.strip()
    parts = prompt.split('_')
    valid_combinations.add((parts[0], parts[1], parts[2]))

# Generate available options
periods = sorted({p for p, _, _ in valid_combinations})
composers = sorted({c for _, c, _ in valid_combinations})
instruments = sorted({i for _, _, i in valid_combinations})

# Global stop flag and thread reference
stop_flag = False
generation_thread = None

# Dynamic component updates
def update_components(period, composer):
    if not period:
        return [
            gr.Dropdown(choices=[], value=None, interactive=False),
            gr.Dropdown(choices=[], value=None, interactive=False)
        ]

    valid_composers = sorted({c for p, c, _ in valid_combinations if p == period})
    valid_instruments = sorted(
        {i for p, c, i in valid_combinations if p == period and c == composer}) if composer else []

    return [
        gr.Dropdown(
            choices=valid_composers,
            value=composer if composer in valid_composers else None,
            interactive=True
        ),
        gr.Dropdown(
            choices=valid_instruments,
            value=None,
            interactive=bool(valid_instruments)
        )
    ]


class RealtimeStream(TextIOBase):
    def __init__(self, queue):
        self.queue = queue

    def write(self, text):
        self.queue.put(text)
        return len(text)


def save_and_convert(abc_content, period, composer, instrumentation):
    if not all([period, composer, instrumentation]):
        raise gr.Error("Please complete a valid generation first before saving")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    prompt_str = f"{period}_{composer}_{instrumentation}"
    filename_base = f"{timestamp}_{prompt_str}"

    abc_filename = f"{filename_base}.abc"
    with open(abc_filename, "w", encoding="utf-8") as f:
        f.write(abc_content)

    xml_filename = f"{filename_base}.xml"
    try:
        subprocess.run(
            ["python", "abc2xml.py", '-o', '.', abc_filename, ],
            check=True,
            capture_output=True,
            text=True
        )
    except subprocess.CalledProcessError as e:
        error_msg = f"Conversion failed: {e.stderr}" if e.stderr else "Unknown error"
        raise gr.Error(f"ABC to XML conversion failed: {error_msg}. Please try to generate another composition.")

    return f"Saved successfully: {abc_filename} -> {xml_filename}"


def generate_music(period, composer, instrumentation, num_bars, metadata_K, metadata_M, model_name, seed, top_k, top_p, temperature):
    global stop_flag, generation_thread
    stop_flag = False  # Reset stop flag when starting

    # Stop any previous thread
    if generation_thread and generation_thread.is_alive():
        stop_flag = True  # Signal the previous thread to stop
        generation_thread.join()  # Wait for the previous thread to stop before starting a new one
        stop_flag = False  # Reset stop flag for new generation

    if (period, composer, instrumentation) not in valid_combinations:
        raise gr.Error("Invalid prompt combination! Please re-select from the period options")

    if model_name not in [entry.name for entry in os.scandir("models") if entry.is_file() and entry.name != "NONE"]:
        raise gr.Error("Invalid Model Name! Please re-select from the config options")

    output_queue = queue.Queue()
    original_stdout = sys.stdout
    sys.stdout = RealtimeStream(output_queue)

    result_container = []

    def run_inference():
        try:
            result_container.append(inference_patch(
                period, composer, instrumentation, num_bars, metadata_K, metadata_M,
                f"models/{model_name}", seed, top_k, top_p, temperature
            ))
        finally:
            sys.stdout = original_stdout

    # Start a new thread for generation
    generation_thread = threading.Thread(target=run_inference)
    generation_thread.start()

    process_output = ""
    while generation_thread.is_alive():
        if stop_flag:  # Stop if requested
            break
        try:
            text = output_queue.get(timeout=0.1)
            process_output += text
            yield process_output, None
        except queue.Empty:
            continue

    while not output_queue.empty():
        text = output_queue.get()
        process_output += text
        yield process_output, None

    final_result = result_container[0] if result_container else ""
    yield process_output, final_result

def stop_generation():
    global stop_flag
    stop_flag = True

with gr.Blocks() as demo:
    gr.Markdown("## NotaGen")

    with gr.Row():
        # 左侧栏
        with gr.Column():
            period_dd = gr.Dropdown(
                choices=periods,
                value=None,
                label="Period",
                interactive=True
            )
            composer_dd = gr.Dropdown(
                choices=[],
                value=None,
                label="Composer",
                interactive=False
            )
            instrument_dd = gr.Dropdown(
                choices=[],
                value=None,
                label="Instrumentation",
                interactive=False
            )

            # Add a collapsible section for configuration parameters
            with gr.Accordion("Config Parameters", open=False):
                models = [entry.name for entry in os.scandir("models") if entry.is_file() and entry.name != "NONE"]
                model_name = gr.Dropdown(
                    choices=models,
                    label="Model Name",
                    value=models[0],
                )
                seed = gr.Slider(minimum=-1, maximum=100000000, step=1, value=-1, label="Seed", info="For Random Seed, Enter -1 (minimum value).")
                top_k = gr.Slider(minimum=1, maximum=20, value=config.TOP_K, label="Top K")
                top_p = gr.Slider(minimum=0.1, maximum=1.0, value=config.TOP_P, label="Top P")
                temperature = gr.Slider(minimum=0.1, maximum=2.0, value=config.TEMPERATURE, label="Temperature")

            # Add a new collapsible section for tune parameters
            with gr.Accordion("Tune Parameters", open=False):
                num_bars = gr.Number(minimum=0, precision=0, label="Number of Bars", value=0, info="For letting the Model to choose the Number of Bars, enter 0.")
                metadata_K = gr.Dropdown(
                    choices=["None", "C", "G", "D", "A", "E", "B", "F#", "C#", "F", "Bb", "Eb", "Ab", "Db", "Gb", "Cb"],
                    label="Key Signature",
                    value="None",
                )
                metadata_M = gr.Textbox(  # todo better handling, check for validity / dropdown
                    label="Time Signature",
                    value="",
                )

            generate_btn = gr.Button("Generate!", variant="primary")
            stop_btn = gr.Button("Stop Generating", visible=False)

            process_output = gr.Textbox(
                label="Generation process",
                interactive=False,
                lines=15,
                max_lines=15,
                placeholder="Generation progress will be shown here...",
                elem_classes="process-output"
            )

        # 右侧栏
        with gr.Column():
            final_output = gr.Textbox(
                label="Post-processed ABC notation scores",
                interactive=True,
                lines=23,
                placeholder="Post-processed ABC scores will be shown here...",
                elem_classes="final-output"
            )

            with gr.Row():
                save_btn = gr.Button("💾 Save as ABC & XML files", variant="secondary")

            save_status = gr.Textbox(
                label="Save Status",
                interactive=False,
                visible=True,
                max_lines=2
            )

    period_dd.change(
        update_components,
        inputs=[period_dd, composer_dd],
        outputs=[composer_dd, instrument_dd]
    )
    composer_dd.change(
        update_components,
        inputs=[period_dd, composer_dd],
        outputs=[composer_dd, instrument_dd]
    )

    generate_btn.click(
        lambda: gr.update(visible=False), outputs=[generate_btn]
    ).then(
        lambda: gr.update(visible=True), outputs=[stop_btn]
    ).then(
        generate_music,
        inputs=[period_dd, composer_dd, instrument_dd, num_bars, metadata_K, metadata_M, model_name, seed, top_k, top_p,
                temperature],
        outputs=[process_output, final_output]
    ).then(
        lambda: gr.update(visible=False), outputs=[stop_btn]
    ).then(
        lambda: gr.update(visible=True), outputs=[generate_btn]
    )

    stop_btn.click(
        stop_generation, inputs=[], outputs=[]
    ).then(
        lambda: gr.update(visible=False), outputs=[stop_btn]
    ).then(
        lambda: gr.update(visible=True), outputs=[generate_btn]
    )

    save_btn.click(
        save_and_convert,
        inputs=[final_output, period_dd, composer_dd, instrument_dd],
        outputs=[save_status]
    )

css = """
.process-output {
    background-color: #f0f0f0;
    font-family: monospace;
    padding: 10px;
    border-radius: 5px;
}
.final-output {
    background-color: #ffffff;
    font-family: sans-serif;
    padding: 10px;
    border-radius: 5px;
}

.process-output textarea {
    max-height: 500px !important;
    overflow-y: auto !important;
    white-space: pre-wrap;
}

"""
css += """
button#💾-save-convert:hover {
    background-color: #ffe6e6;
}
"""

demo.css = css

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=True
    )