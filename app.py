import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-instruct")  # Or your own!
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-1.3b-instruct", torch_dtype=torch.float32, trust_remote_code=True)

def generate_code(prompt, style="Clean & Pythonic"):
    if style == "Verbose like a 15th-century manuscript":
        prompt = "In a manner most detailed, write code that... " + prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, 
                             max_new_tokens=256,
                            do_sample=True,
                            temperature=1.0,
                            top_p=0.95,
                            use_cache=False)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

demo = gr.Interface(
    fn=generate_code,
    inputs=[
        gr.Textbox(label="How shall Codice Da Vinci help today?", lines=3),
        gr.Dropdown(["Clean & Pythonic", "Verbose like a 15th-century manuscript"], label="Code Style")
    ],
    outputs=gr.Code(label="🧾 Leonardo's Work"),
    title="Codice Da Vinci 📜💻",
    description="Your Renaissance coding assistant. Fluent in algorithms and Latin. Powered by LLM."
)

demo.launch()
