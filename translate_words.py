#!/usr/bin/env python3
"""
translate_words.py  –  Translate Chinese vocabulary to English.

Modes
-----
contextual  (default)
    Uses facebook/nllb-200-distilled-1.3B (seq2seq MT model).
    Fast batch translation; output mirrors how the word is used in context.

dictionary
    Uses an LLM (Qwen2.5-3B-Instruct by default, overridable via HF_MODEL_ID).
    Prompts the model for a concise dictionary entry, e.g. "to bite; to chew".
    Slower (one word at a time) but produces cleaner flashcard-style definitions.

Usage
-----
    python translate_words.py                   # contextual mode
    python translate_words.py --mode dictionary # dictionary mode
"""
import argparse
import os
import warnings

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Contextual mode (NLLB-200)
# ---------------------------------------------------------------------------

def _load_nllb(device: torch.device):
    model_name = "facebook/nllb-200-distilled-1.3B"
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    model.eval()
    print("Model loaded successfully!")
    return tokenizer, model


def _translate_contextual(words: list[str], tokenizer, model, device: torch.device) -> list[str]:
    """Batch-translate words using NLLB-200 (contextual MT)."""
    src_lang = "zho_Hant"  # Chinese Traditional
    tgt_lang = "eng_Latn"  # English
    tokenizer.src_lang = src_lang
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)

    translations: list[str] = []
    batch_size = 32
    max_length = 32

    for i in tqdm(range(0, len(words), batch_size), desc="Translating (contextual)"):
        batch = words[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=max_length).to(device)
        with torch.no_grad():
            translated = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=max_length,
                num_beams=5,
                early_stopping=True,
            )
        translations.extend(tokenizer.batch_decode(translated, skip_special_tokens=True))

    return translations


# ---------------------------------------------------------------------------
# Dictionary mode (LLM prompt)
# ---------------------------------------------------------------------------

def _load_llm(device: torch.device):
    model_name = os.getenv("HF_MODEL_ID", "Qwen/Qwen2.5-3B-Instruct")
    print(f"Loading LLM: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if not torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="cpu", torch_dtype=torch.float32, low_cpu_mem_usage=True
        )
    else:
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb, device_map="auto",
            torch_dtype=torch.float16, low_cpu_mem_usage=True,
        )

    model.eval()
    print("LLM loaded successfully!")
    return tokenizer, model


def _translate_dictionary_one(word: str, tokenizer, model) -> str:
    """Ask the LLM for a concise dictionary definition of a single word."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a Chinese–English dictionary. "
                "When given a Chinese word, reply with only its concise English dictionary entry: "
                "part of speech (optional) and the main meaning(s) separated by semicolons. "
                "No explanations, no example sentences, no Chinese characters in the answer."
            ),
        },
        {
            "role": "user",
            "content": f"Chinese word: {word}",
        },
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=False,          # greedy – deterministic dictionary entries
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0, inputs["input_ids"].shape[-1]:]
    result = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    # Keep only the first line in case the model adds extra text
    return result.splitlines()[0].strip()


def _translate_dictionary(words: list[str], tokenizer, model) -> list[str]:
    translations: list[str] = []
    for word in tqdm(words, desc="Translating (dictionary)"):
        translations.append(_translate_dictionary_one(word, tokenizer, model))
    return translations


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def translate_words(mode: str = "contextual") -> None:
    print(f"Mode: {mode}")
    print("Reading words.csv...")
    df = pd.read_csv('data/words.csv')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    words_to_translate = df['Vocabulary'].tolist()
    print(f"Words to translate: {len(words_to_translate)}")

    if mode == "dictionary":
        tokenizer, model = _load_llm(device)
        translations = _translate_dictionary(words_to_translate, tokenizer, model)
    else:
        tokenizer, model = _load_nllb(device)
        translations = _translate_contextual(words_to_translate, tokenizer, model, device)

    df['trans'] = translations

    output_file = 'data/words_final.csv'
    df.to_csv(output_file, index=False)
    print(f"\nTranslation complete! Saved to {output_file}")
    print(f"Total words translated: {len(translations)}")

    print("\nExample translations:")
    for i in range(min(5, len(df))):
        print(f"  {df.iloc[i]['Vocabulary']:10s} -> {df.iloc[i]['trans']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate Chinese vocabulary to English.")
    parser.add_argument(
        "--mode",
        choices=["contextual", "dictionary"],
        default="contextual",
        help="'contextual' uses NLLB-200 (fast, MT-style); "
             "'dictionary' uses an LLM to produce concise dictionary entries.",
    )
    args = parser.parse_args()
    translate_words(mode=args.mode)
