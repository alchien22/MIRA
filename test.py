from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

model_name = "johnsnowlabs/JSL-MedLlama-3-8B-v2.0"

quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)  # Enable 8-bit quantization
# quant_config = BitsAndBytesConfig(load_in_8bit=True)

try:
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map="auto",
        quantization_config=quant_config, 
        output_hidden_states=True,
        return_dict_in_generate=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.chat_template = """
{% for message in messages %}
{% if loop.first %}
<|begin_of_text|>
{% endif %}
{% if message['role'] == 'system' %}
<|start_header_id|>system<|end_header_id|>
{{ message['content'] }}<|eot_id|>
{% elif message['role'] == 'user' %}
<|start_header_id|>user<|end_header_id|>
{% if message['context'] %}
Context:
{{ message['context'] }}
{% endif %}

Question:
{{ message['content'] }}<|eot_id|>
{% elif message['role'] == 'assistant' %}
<|start_header_id|>assistant<|end_header_id|>
{{ message['content'] }}<|eot_id|>
{% endif %}
{% if loop.last and add_generation_prompt %}
<|start_header_id|>assistant<|end_header_id|>
{% endif %}
{% endfor %}
    """
    print("✅ Model loaded successfully in 4-bit mode!")
except Exception as e:
    print("❌ Model could not be loaded in 4-bit mode:", str(e))

chat = [
    {"role": "system", "content": "You are a medical expert AI assistant called MIRA. Provide concise responses."},
    {
        "role": "user",
        "context": "",
        "content": "What is hodgkins lymphoma?"
    },
]

prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
print(prompt)
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
output = model.generate(
    **inputs, 
    max_new_tokens=200, 
    return_dict_in_generate=True,
    output_hidden_states=True,
    output_scores=True,
    pad_token_id=tokenizer.eos_token_id
)
generated_tokens = output.sequences[0][inputs['input_ids'].shape[-1]:]
response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
print(response)