import concurrent.futures
from flask import Flask, request, render_template, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "./llama_finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

app = Flask(__name__)

executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)  # Run the model inference in a separate thread

def generate_code_async(prompt, temperature, max_length):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
    attention_mask = inputs.get("attention_mask", torch.ones_like(inputs["input_ids"]))
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=attention_mask,
            max_length=max_length,
            temperature=temperature,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            top_k=50,
            top_p=0.95
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    try:
        prompt = request.form["prompt"]
        temperature = float(request.form.get("temperature", 0.7))
        max_length = int(request.form.get("max_length", 200))

        future = executor.submit(generate_code_async, prompt, temperature, max_length)
        generated_code = future.result()

        return render_template("index.html", prompt=prompt, generated_code=generated_code)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
