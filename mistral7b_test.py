import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, QuantoConfig
# import quanto

model_name = "mistralai/Mistral-7B-Instruct-v0.3"

# Check for CUDA, then MPS, then CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")


tokenizer = AutoTokenizer.from_pretrained(model_name)
# quantization_config = QuantoConfig(weights="int4") # This often fails


model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16, # Load in float16 initially
    # quantization_config=quantization_config, # Apply Quanto quantization
    device_map="auto" # Let HF Accelerate handle initial device mapping
)
model.to(device) 


prompt = "What is the capital of New Hampshire?"
chat_template_prompt = f"<s>[INST] {prompt.strip()} [/INST]"
inputs = tokenizer(chat_template_prompt, return_tensors="pt").to(device)


print("Generating response...")
print("Generating response...")
try:
    outputs = model.generate(**inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response_only = response_text.split("[/INST]")[-1].strip()
    print(f"Prompt: {prompt}")
    print(f"Response: {response_only}")

except Exception as e:
    print(f"Error during generation: {e}")
    import traceback
    traceback.print_exc()
