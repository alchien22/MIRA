from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_name = "BioMistral/BioMistral-7B"

quant_config = BitsAndBytesConfig(load_in_8bit=True)  # Enable 8-bit quantization

try:
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", quantization_config=quant_config)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("✅ Model loaded successfully in 8-bit mode!")
except Exception as e:
    print("❌ Model could not be loaded in 8-bit mode:", str(e))


inputs = tokenizer("What are the symptoms of diabetes?", return_tensors="pt").to("cuda")
output = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(output[0], skip_special_tokens=True))