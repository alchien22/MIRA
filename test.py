from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_name = "ProbeMedicalYonseiMAILab/medllama3-v20"

quant_config = BitsAndBytesConfig(load_in_8bit=True)  # Enable 8-bit quantization

try:
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", quantization_config=quant_config)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("✅ Model loaded successfully in 8-bit mode!")
except Exception as e:
    print("❌ Model could not be loaded in 8-bit mode:", str(e))
