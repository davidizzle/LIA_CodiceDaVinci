import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct
model_id = "deepseek-ai/deepseek-coder-6.7b-instruct"
# model_id = "deepseek-ai/deepseek-coder-33b-instruct"
# model_id = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
# model_id = "deepseek-ai/DeepSeek-Coder-V2-Instruct"

# This works best
# model_id = "deepseek-ai/deepseek-coder-1.3b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, 
                                            #  device_map=None, 
                                            #  torch_dtype=torch.float32, 
                                             device_map="auto", 
                                             torch_dtype=torch.float16, 
                                             trust_remote_code=True
                                             )
# model.to("cpu")

def generate_code(prompt, style="Clean & Pythonic"):
    # spinner.update(visible=True)
    if style == "Verbose like a 15th-century manuscript":
        prompt = "In a manner most detailed, write code that... " + prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, 
                            #  max_new_tokens=100,
                            max_new_tokens=512,
                            do_sample=False,
                            temperature=1.0,
                            top_p=0.95,
                            top_k=50, 
                            num_return_sequences=1, 
                            eos_token_id=tokenizer.eos_token_id
                            )
    # spinner.update(visible=False)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

with gr.Blocks() as demo:
    gr.Markdown("<h1 style='text-align:center;'>Codice Da Vinci 📜💻</h1>")

    with gr.Row():
        prompt = gr.Textbox(label="How shall Codice Da Vinci help today?", lines=3)
        style = gr.Dropdown(["Clean & Pythonic", "Verbose like a 15th-century manuscript"], label="Code Style")

    generate_btn = gr.Button("Generate")


    output = gr.Code(label="🧾 Leonardo's Work")

    spinner = gr.HTML(
        "<div style='text-align:center'><img src='https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExMXViMm02MnR6bGJ4c2h3ajYzdWNtNXNtYnNic3lnN2xyZzlzbm9seSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9cw/k32ddF9WVs44OUaZAm/giphy.gif' width='80'></div>",
        visible=False
    )
    
    def wrapped_generate(prompt, style):
        code = generate_code(prompt, style)
        return gr.update(visible=False), code

    generate_btn.click(
        fn=wrapped_generate,
        inputs=[prompt, style],
        outputs=[spinner, output],
        show_progress=True,
        preprocess=True
    )

    generate_btn.click(
        fn=lambda: gr.update(visible=True),
        inputs=[],
        outputs=[spinner],
        queue=False
    )

demo.launch()