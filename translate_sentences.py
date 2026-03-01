import csv
import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Initialize model and tokenizer (using NLLB-200 3.3B - great for Chinese-English)
MODEL_NAME = "facebook/nllb-200-3.3B"
print(f"Loading model {MODEL_NAME}...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto"
)
print("Model loaded successfully!")

def translate_text(text):
    """Translate a single Chinese text to English using NLLB-200"""
    # NLLB language codes: zho_Hans (Simplified Chinese), zho_Hant (Traditional Chinese), eng_Latn (English)
    tokenizer.src_lang = "zho_Hant"  # Traditional Chinese
    
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate translation
    translated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id["eng_Latn"],
        max_length=512,
        num_beams=5,
        early_stopping=True
    )
    
    translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    return translation.strip()

def main():
    input_file = 'sentences.csv'
    
    # Read the CSV file
    print(f"Reading {input_file}...")
    rows = []
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            rows.append(row)
    
    print(f"Found {len(rows)} rows")
    
    # Find sentences that need translation
    to_translate = []
    for i, row in enumerate(rows):
        sentence = row.get('sentence', '') or ''
        sentence = sentence.strip()
        translation = row.get('Translation', '') or ''
        translation = translation.strip()
        
        # Skip if sentence contains [FAILED] or already has translation
        if '[FAILED]' not in sentence and sentence and not translation:
            to_translate.append(i)
    
    print(f"Found {len(to_translate)} sentences to translate")
    
    if not to_translate:
        print("No sentences need translation!")
        return
    
    # Translate sentences with batch processing
    batch_size = 50  # Can process more since we're using local GPU
    translated_count = 0
    
    for idx in to_translate[:batch_size]:
        sentence = rows[idx]['sentence']
        print(f"\nTranslating ({translated_count+1}/{min(batch_size, len(to_translate))}): {sentence}")
        
        try:
            translation = translate_text(sentence)
            rows[idx]['Translation'] = translation
            translated_count += 1
            print(f"Translation: {translation}")
        except Exception as e:
            print(f"Error translating: {e}")
            import traceback
            traceback.print_exc()
    
    # Write back to file
    if translated_count > 0:
        print(f"\nWriting {translated_count} translations back to {input_file}...")
        with open(input_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['word', 'sentence', 'Translation'], delimiter=';')
            writer.writeheader()
            writer.writerows(rows)
        print("Done!")
    else:
        print("No translations were added.")

if __name__ == "__main__":
    main()