from transformers import AutoModelForCausalLM

model_name = "ProbeMedicalYonseiMAILab/medllama3-v20"
model = AutoModelForCausalLM.from_pretrained(model_name)
print(model.dtype)  # If it's float16 or float32, 8-bit should work
