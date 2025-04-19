import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct
model_id = "deepseek-ai/deepseek-coder-1.3b-instruct"
# model_id = "deepseek-ai/deepseek-coder-6.7b-instruct"
# model_id = "deepseek-ai/deepseek-coder-33b-instruct"
# model_id = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
# model_id = "deepseek-ai/DeepSeek-Coder-V2-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)  # Or your own!
model = AutoModelForCausalLM.from_pretrained(model_id, 
                                            #  device_map=None, 
                                            #  torch_dtype=torch.float32, 
                                             device_map="auto", 
                                             torch_dtype=torch.float16, 
                                             trust_remote_code=True
                                             )
# model.to("cpu")

def generate_code(prompt, style="Clean & Pythonic"):
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
