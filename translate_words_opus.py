#!/usr/bin/env python3
import pandas as pd
import torch
from transformers import MarianMTModel, MarianTokenizer
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
    
    # Load Opus-MT Chinese to English model
    model_name = "Helsinki-NLP/opus-mt-zh-en"
    print(f"Loading model: {model_name}")
    
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name).to(device)
    model.eval()
    
    print("Model loaded successfully!")
    
    # Extract vocabulary words for translation
    words_to_translate = df['Vocabulary'].tolist()
    print(f"\nTranslating {len(words_to_translate)} words...")
    
    # Translate in batches
    def translate_batch(texts, batch_size=64, max_length=64):
        """Translate Chinese words to English using Opus-MT."""
        translations = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Translating"):
            batch = texts[i:i + batch_size]
            
            # Tokenize
            inputs = tokenizer(batch, return_tensors="pt", padding=True, 
                              truncation=True, max_length=max_length).to(device)
            
            # Generate translations
            with torch.no_grad():
                translated = model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=4,
                    early_stopping=True
                )
            
            # Decode translations
            batch_translations = tokenizer.batch_decode(translated, skip_special_tokens=True)
            translations.extend(batch_translations)
        
        return translations
    
    # Translate all words
    translations = translate_batch(words_to_translate, batch_size=64)
    
    # Add the translations column
    df['trans'] = translations
    
    # Save to new file
    output_file = 'data/words_final.csv'
    df.to_csv(output_file, index=False)
    print(f"\nTranslation complete! Saved to {output_file}")
    print(f"Total words translated: {len(translations)}")
    
    # Show some examples
    print("\nExample translations:")
    for i in range(min(10, len(df))):
        print(f"{df.iloc[i]['Vocabulary']:10s} -> {df.iloc[i]['trans']}")

if __name__ == "__main__":
    translate_words()
