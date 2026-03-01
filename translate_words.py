#!/usr/bin/env python3
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

def translate_words():
    # Read the CSV file
    print("Reading words.csv...")
    df = pd.read_csv('data/words.csv')
    
    # Check GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Load pre-trained NLLB-200 model for Chinese to English translation
    model_name = "facebook/nllb-200-distilled-1.3B"
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    model.eval()  # Set to evaluation mode
    
    # NLLB language codes
    src_lang = "zho_Hant"  # Chinese Traditional
    tgt_lang = "eng_Latn"  # English
    
    print("Model loaded successfully!")
    print(f"Source language: {src_lang} -> Target language: {tgt_lang}")
    
    # Extract vocabulary words for translation
    words_to_translate = df['Vocabulary'].tolist()
    print(f"\nTranslating {len(words_to_translate)} words...")
    
    # Translate in batches with shorter max_length to get concise translations
    def translate_batch(texts, batch_size=32, max_length=32):
        """Translate a list of Chinese words to English using GPU acceleration."""
        translations = []
        
        # Set source language
        tokenizer.src_lang = src_lang
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Translating"):
            batch = texts[i:i + batch_size]
            
            # Tokenize the batch
            inputs = tokenizer(batch, return_tensors="pt", padding=True, 
                              truncation=True, max_length=max_length).to(device)
            
            # Generate translations with target language code
            forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)
            with torch.no_grad():
                translated = model.generate(
                    **inputs, 
                    forced_bos_token_id=forced_bos_token_id,
                    max_length=max_length,
                    num_beams=5,  # Use beam search for better quality
                    early_stopping=True
                )
            
            # Decode translations
            batch_translations = tokenizer.batch_decode(translated, skip_special_tokens=True)
            translations.extend(batch_translations)
        
        return translations
    
    # Translate all words
    translations = translate_batch(words_to_translate, batch_size=32)
    
    # Add the translations column
    df['trans'] = translations
    
    # Save to new file
    output_file = 'data/words_final.csv'
    df.to_csv(output_file, index=False)
    print(f"\nTranslation complete! Saved to {output_file}")
    print(f"Total words translated: {len(translations)}")
    
    # Show some examples
    print("\nExample translations:")
    for i in range(min(5, len(df))):
        print(f"{df.iloc[i]['Vocabulary']} -> {df.iloc[i]['trans']}")

if __name__ == "__main__":
    translate_words()
