# translator.py - Patched version to fix past_key_values bug
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from IndicTransToolkit.processor import IndicProcessor
import torch
import unicodedata
import warnings

# Monkey patch to fix the IndicTrans2 bug
def patch_indictrans_model(model):
    """
    Monkey patch to fix the past_key_values bug in IndicTrans2 models
    """
    original_forward = model.model.decoder.forward
    
    def patched_forward(self, *args, **kwargs):
        # If past_key_values is passed and contains None values, remove it
        if 'past_key_values' in kwargs:
            past_kv = kwargs['past_key_values']
            if past_kv is not None:
                # Check if any layer has None values
                has_none = False
                for layer_kv in past_kv:
                    if layer_kv is None or any(x is None for x in layer_kv):
                        has_none = True
                        break
                
                if has_none:
                    kwargs['past_key_values'] = None
        
        return original_forward(*args, **kwargs)
    
    # Replace the forward method
    model.model.decoder.forward = patched_forward.__get__(model.model.decoder, model.model.decoder.__class__)
    return model

class IndicPivotTranslator:
    """
    Translator that uses transliteration as a pivot:
      Romanized (ITRANS) <-> Devanagari <-> English
    """

    def __init__(self, 
                 en_indic_model="ai4bharat/indictrans2-en-indic-dist-200M",
                 indic_en_model="ai4bharat/indictrans2-indic-en-dist-200M", 
                 device=None):
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load English -> Indic model
        print("Loading English -> Indic model...")
        self.en_indic_tokenizer = AutoTokenizer.from_pretrained(
            en_indic_model, 
            use_fast=False, 
            trust_remote_code=True
        )
        self.en_indic_model = AutoModelForSeq2SeqLM.from_pretrained(
            en_indic_model, 
            trust_remote_code=True,
            dtype=torch.float32  # Use float32 to avoid precision issues
        )
        
        # Patch the model to fix the bug
        self.en_indic_model = patch_indictrans_model(self.en_indic_model)
        self.en_indic_model.to(self.device)
        
        # Load Indic -> English model  
        print("Loading Indic -> English model...")
        self.indic_en_tokenizer = AutoTokenizer.from_pretrained(
            indic_en_model, 
            use_fast=False, 
            trust_remote_code=True
        )
        self.indic_en_model = AutoModelForSeq2SeqLM.from_pretrained(
            indic_en_model, 
            trust_remote_code=True,
            dtype=torch.float32
        )
        
        # Patch this model too
        self.indic_en_model = patch_indictrans_model(self.indic_en_model)
        self.indic_en_model.to(self.device)
        
        # Set pad tokens if not already set
        if self.en_indic_tokenizer.pad_token is None:
            self.en_indic_tokenizer.pad_token = self.en_indic_tokenizer.eos_token
        if self.indic_en_tokenizer.pad_token is None:
            self.indic_en_tokenizer.pad_token = self.indic_en_tokenizer.eos_token
        
        # Processor for both directions
        self.ip = IndicProcessor(inference=True)

    # ----- transliteration helpers -----
    def roman_to_dev(self, roman_text: str) -> str:
        try:
            return transliterate(roman_text, sanscript.ITRANS, sanscript.DEVANAGARI)
        except Exception as e:
            print(f"Error in roman_to_dev: {e}")
            return roman_text

    def dev_to_roman(self, dev_text: str) -> str:
        try:
            if not isinstance(dev_text, str):
                dev_text = str(dev_text)
            return transliterate(dev_text, sanscript.DEVANAGARI, sanscript.ITRANS)
        except Exception as e:
            print(f"Error in dev_to_roman: {e}")
            return dev_text

    def dev_to_english(self, dev_text: str, max_length=128) -> str:
        """Translate Devanagari text to English"""
        try:
            src_lang, tgt_lang = "hin_Deva", "eng_Latn"
            
            # Preprocess the input
            batch = self.ip.preprocess_batch([dev_text], src_lang=src_lang, tgt_lang=tgt_lang)
            
            # Tokenize
            inputs = self.indic_en_tokenizer(
                batch, 
                truncation=True, 
                return_tensors="pt", 
                padding=True,
                max_length=256
            ).to(self.device)
                        
            # Generate with minimal parameters to avoid the bug
            with torch.no_grad():
                outputs = self.indic_en_model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=max_length,
                    do_sample=False,
                    use_cache=False,  # Disable caching to avoid past_key_values issues
                    pad_token_id=self.indic_en_tokenizer.pad_token_id,
                    eos_token_id=self.indic_en_tokenizer.eos_token_id,
                    temperature=1.0,
                    top_p=1.0
                )
                        
            # Decode the output
            generated_tokens = self.indic_en_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Postprocess
            result = self.ip.postprocess_batch([generated_tokens], lang=tgt_lang)[0]
            return result
            
        except Exception as e:
            print(f"Error in dev_to_english: {e}")
            import traceback
            traceback.print_exc()
            return f"Translation error: {dev_text}"

    def english_to_dev(self, english_text: str, max_length=128) -> str:
        """Translate English text to Devanagari"""
        try:
            src_lang, tgt_lang = "eng_Latn", "hin_Deva"
            
            # Preprocess the input
            batch = self.ip.preprocess_batch([english_text], src_lang=src_lang, tgt_lang=tgt_lang)
            
            # Tokenize
            inputs = self.en_indic_tokenizer(
                batch, 
                truncation=True, 
                return_tensors="pt", 
                padding=True,
                max_length=256
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.en_indic_model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=max_length,
                    do_sample=False,
                    use_cache=False,  # Disable caching
                    pad_token_id=self.en_indic_tokenizer.pad_token_id,
                    eos_token_id=self.en_indic_tokenizer.eos_token_id
                )
            
            # Decode the output
            generated_tokens = self.en_indic_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Postprocess
            result = self.ip.postprocess_batch([generated_tokens], lang=tgt_lang)[0]
            return result
            
        except Exception as e:
            print(f"Error in english_to_dev: {e}")
            import traceback
            traceback.print_exc()
            return f"Translation error: {english_text}"

    # ----- public API for Romanized <-> English using Devanagari pivot -----
    def roman_to_english(self, roman_text: str, max_length=128) -> str:
        """Convert romanized text to English via Devanagari pivot"""
        try:
            # Step 1: Roman -> Devanagari
            dev_text = self.roman_to_dev(roman_text)
            dev_text = unicodedata.normalize('NFC', dev_text)
            
            # Step 2: Devanagari -> English
            english_text = self.dev_to_english(dev_text, max_length=max_length)
            
            return english_text
        except Exception as e:
            print(f"Error in roman_to_english: {e}")
            return f"Translation error: {roman_text}"

    def english_to_roman(self, english_text: str, max_length=128) -> str:
        """Convert English text to romanized via Devanagari pivot"""
        try:
            # Step 1: English -> Devanagari
            dev_text = self.english_to_dev(english_text, max_length=max_length)

            # Step 2: Devanagari -> Roman
            roman_text = self.dev_to_roman(dev_text)   
            
            return roman_text
        except Exception as e:
            print(f"Error in english_to_roman: {e}")
            return f"Translation error: {english_text}"

# Test code
if __name__ == "__main__":
    try:
        print("\n=== Testing Patched Class Approach ===")
        t = IndicPivotTranslator()
        
        print("\n=== Testing Roman to English ===")
        result1 = t.roman_to_english("tum kya kar rahe ho?")
        print(f"Final result: {result1}")
        
        print("\n=== Testing English to Roman ===")
        result2 = t.english_to_roman("your name is Arjun")
        print(f"Final result: {result2}")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()