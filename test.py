from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load the tokenizer and model from Hugging Face
model_name = "meta-llama/Llama-3-7b-hf"  # Example model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

prompt = "Summarize the latest news in Bangladesh:"
result = generator(prompt, max_length=200)
print(result[0]['generated_text'])
