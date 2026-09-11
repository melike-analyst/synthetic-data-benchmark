import os
import json
import time
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

# .env dosyasındaki ANTHROPIC_API_KEY değişkenini okuyup Gemini'ye tanımlıyoruz
api_key = os.getenv("ANTHROPIC_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

def generate_batch(schema_description, example_rows_json, n=20, model="gemini-3.5-flash-lite"):
    prompt = f"""You generate synthetic tabular data that statistically resembles real data.

Schema: {schema_description}

Here are 5 real example rows, shown ONLY so you understand format, value ranges and column relationships.
Do NOT copy these rows. Generate NEW rows with similar statistical patterns:
{example_rows_json}

Return ONLY a valid JSON array of {n} new row objects. No explanation, no markdown formatting, no code fences."""

    # Google Gemini modelini başlatıyoruz
    gemini_model = genai.GenerativeModel(model)
    response = gemini_model.generate_content(
        prompt,
        generation_config={"response_mime_type": "application/json"}
    )
    text = response.text.strip()

    # Model bazen ```json ile sarmalayabilir, temizleyelim
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]

    rows = json.loads(text)
    
    # Token kullanım metriklerini mevcut yapına uygun nesne formatına getiriyoruz
    usage_data = getattr(response, 'usage_metadata', None)
    
    class Usage:
        def __init__(self, inp, outp):
            self.input_tokens = inp
            self.output_tokens = outp

    input_tokens = usage_data.prompt_token_count if usage_data else 0
    output_tokens = usage_data.candidates_token_count if usage_data else 0
    
    usage = Usage(input_tokens, output_tokens)
    return rows, usage


def generate_synthetic_dataset(train_df, schema_description, target_n=500, batch_size=50):
    examples = train_df.sample(5).to_dict(orient='records')
    examples_json = json.dumps(examples, default=str)

    all_rows = []
    total_input_tokens = 0
    total_output_tokens = 0

    n_batches = target_n // batch_size
    start = time.time()

    for i in range(n_batches):
        rows, usage = generate_batch(schema_description, examples_json, n=batch_size)
        all_rows.extend(rows)
        total_input_tokens += usage.input_tokens
        total_output_tokens += usage.output_tokens
        time.sleep(2)  # Rate-limit yaşamamak için 2 saniye bekleme
        print(f"Batch {i+1}/{n_batches} tamamlandı, toplam satır: {len(all_rows)}")

    elapsed = time.time() - start

    return pd.DataFrame(all_rows), {
        "elapsed_sec": elapsed,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens
    }