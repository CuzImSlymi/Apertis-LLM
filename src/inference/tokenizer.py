# /home/ubuntu/ApertisAI_Project/Apertis AI_/src/inference/tokenizer.py
# Implements a tokenizer using the Hugging Face tokenizers library.

import logging
import json
import os
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from typing import Optional, Dict, List # Added Optional, Dict, List

logger = logging.getLogger(__name__)

class HFTokenizer:
    def __init__(self, vocab_file: Optional[str] = None, vocab_dict: Optional[Dict[str, int]] = None):
        """
        Initializes the tokenizer from a vocabulary file or dictionary.

        Args:
            vocab_file: Path to the vocabulary file (JSON format).
                        Can be a tokenizer.json file or a JSON containing a vocab dict.
            vocab_dict: A dictionary mapping tokens (str) to IDs (int).
                        If provided, this takes precedence over vocab_file.
        """
        self.tokenizer = None
        self.vocab_file = vocab_file
        self.vocab_dict_input = vocab_dict # Store the input dict
        if vocab_file is None and vocab_dict is None:
             logger.warning("HFTokenizer initialized without vocab_file or vocab_dict. Will use fallback.")
        self._load_tokenizer()

    def _load_tokenizer(self):
        """Loads or creates the tokenizer based on the provided vocab file or dict."""
        vocab = None
        source_info = "unknown source"
        try:
            # Priority 1: Use provided vocab_dict
            if self.vocab_dict_input is not None:
                if isinstance(self.vocab_dict_input, dict) and \
                   all(isinstance(k, str) and isinstance(v, int) for k, v in self.vocab_dict_input.items()):
                    # Use a copy to avoid modifying the original dict if passed by reference
                    vocab = self.vocab_dict_input.copy()
                    source_info = "provided dictionary"
                    logger.info(f"Using provided vocabulary dictionary with {len(vocab)} tokens.")
                else:
                    logger.warning("Provided vocab_dict is not a valid dictionary {str: int}. Ignoring and attempting to load from file if provided.")

            # Priority 2: Load from vocab_file if vocab not loaded from dict
            if vocab is None and isinstance(self.vocab_file, str):
                source_info = f"file '{self.vocab_file}'"
                logger.info(f"Attempting to load vocabulary from {source_info}")

                # Check if file exists before proceeding
                if not os.path.exists(self.vocab_file):
                    raise FileNotFoundError(f"Vocabulary file not found: {self.vocab_file}")

                # Attempt to load directly if it's a tokenizer.json file saved by tokenizers library
                if self.vocab_file.endswith(".json"): # Keep check broad, could be tokenizer.json or vocab.json
                    try:
                        # Try loading as a full tokenizer config first
                        self.tokenizer = Tokenizer.from_file(self.vocab_file)
                        logger.info(f"Loaded tokenizer directly from file: {self.vocab_file}")
                        # If loaded directly, we are done with loading/building.
                        return
                    except Exception as e:
                        logger.warning(f"Could not load {self.vocab_file} as a full tokenizer file ({e}). Will try parsing as a JSON vocab dictionary.")
                        # Fall through to attempt parsing as a vocab dict JSON

                # If not a directly loadable tokenizer file or direct load failed,
                # assume it's a JSON containing a vocab dictionary {token: id} or {"tokens": [...]} or tokenizer structure
                try:
                    with open(self.vocab_file, "r", encoding='utf-8') as f: # Added encoding
                        loaded_json = json.load(f)

                    # Handle different possible vocab formats within the JSON
                    if isinstance(loaded_json, dict):
                        if "tokens" in loaded_json and isinstance(loaded_json["tokens"], list):
                            # Format: {"tokens": ["token1", "token2", ...]}
                            token_list = loaded_json["tokens"]
                            vocab = {token: idx for idx, token in enumerate(token_list)}
                            logger.info(f"Built vocab from token list in {self.vocab_file}")
                        elif all(isinstance(k, str) and isinstance(v, int) for k, v in loaded_json.items()):
                            # Standard format: {"token1": 0, "token2": 1, ...}
                            vocab = loaded_json
                            logger.info(f"Using vocab dict from {self.vocab_file}")
                        else:
                            # Check if it might be a tokenizer.json structure we couldn't load directly
                            if "model" in loaded_json and "vocab" in loaded_json["model"] and isinstance(loaded_json["model"]["vocab"], dict):
                                logger.info(f"Extracting vocab dict from 'model.vocab' structure in {self.vocab_file}")
                                vocab = loaded_json["model"]["vocab"]
                                # Ensure values are integers
                                vocab = {k: int(v) for k, v in vocab.items() if isinstance(v, (int, float, str)) and str(v).isdigit()}
                            else:
                                raise ValueError("Unsupported vocabulary dictionary format in JSON file.")
                    else:
                        raise ValueError("Vocabulary file does not contain a JSON dictionary.")
                except json.JSONDecodeError:
                    logger.error(f"Failed to decode JSON from vocabulary file: {self.vocab_file}")
                    raise # Re-raise JSONDecodeError to trigger fallback
                except Exception as e: # Catch other potential errors during file processing
                    logger.error(f"Error processing vocabulary file {self.vocab_file}: {e}")
                    raise # Re-raise to trigger fallback

            # If vocab is still None after trying dict and file, it means neither was provided or valid.
            if vocab is None:
                 logger.warning("No valid vocabulary source found (dict or file). Proceeding with minimal fallback.")
                 raise ValueError("No valid vocabulary source.") # Raise error to enter the except block for fallback

            # --- Common logic for building tokenizer from vocab dict ---
            # Define special tokens (use defaults if not found)
            unk_token = "<unk>"
            bos_token = "<bos>"
            eos_token = "<eos>"
            pad_token = "<pad>"

            # Ensure special tokens are in vocab, add them if necessary
            next_id = 0
            if vocab: # Find next available ID only if vocab is not empty
                try:
                    # Ensure all values are integers before finding max
                    int_values = [v for v in vocab.values() if isinstance(v, int)]
                    if int_values:
                        next_id = max(int_values) + 1
                except Exception as e:
                    logger.error(f"Error determining next ID from vocab values: {e}. Resetting next_id to 0.")
                    next_id = 0 # Reset if error occurs

            special_tokens_map = {} # Store added special tokens and their IDs
            added_tokens = [] # Keep track of tokens added

            # Check and add UNK token
            if unk_token not in vocab:
                vocab[unk_token] = next_id
                special_tokens_map["unk_token"] = unk_token
                added_tokens.append(unk_token)
                next_id += 1
            else:
                # Ensure existing unk_token is recorded for WordLevel
                special_tokens_map["unk_token"] = unk_token

            # Check and add BOS token
            if bos_token not in vocab:
                 vocab[bos_token] = next_id
                 special_tokens_map["bos_token"] = bos_token
                 added_tokens.append(bos_token)
                 next_id += 1

            # Check and add EOS token
            if eos_token not in vocab:
                 vocab[eos_token] = next_id
                 special_tokens_map["eos_token"] = eos_token
                 added_tokens.append(eos_token)
                 next_id += 1

            # Check and add PAD token
            if pad_token not in vocab:
                 vocab[pad_token] = next_id
                 special_tokens_map["pad_token"] = pad_token
                 added_tokens.append(pad_token)
                 next_id += 1

            if added_tokens:
                logger.info(f"Added missing special tokens to vocabulary: {added_tokens}")

            # Create a WordLevel tokenizer using the prepared vocab
            # Pass the determined unk_token to the model
            self.tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token=special_tokens_map["unk_token"]))

            # Use basic whitespace pre-tokenization
            self.tokenizer.pre_tokenizer = Whitespace()

            # Define post-processing to add BOS/EOS tokens if they exist in the final vocab
            bos_token_id = vocab.get(bos_token)
            eos_token_id = vocab.get(eos_token)

            if bos_token_id is not None and eos_token_id is not None:
                self.tokenizer.post_processor = TemplateProcessing(
                    single=f"{bos_token} $A {eos_token}",
                    # pair=f"{bos_token} $A {eos_token} $B:1 {eos_token}:1", # Example for pairs if needed
                    special_tokens=[(bos_token, bos_token_id), (eos_token, eos_token_id)],
                )
                logger.info("Configured post-processor to add BOS/EOS tokens.")
            else:
                logger.warning("BOS or EOS token not found in final vocabulary. Post-processor not configured.")

            # Enable padding if PAD token exists
            pad_token_id = vocab.get(pad_token)
            if pad_token_id is not None:
                 self.tokenizer.enable_padding(pad_id=pad_token_id, pad_token=pad_token)
                 logger.info("Enabled padding with PAD token.")
            else:
                 logger.warning("PAD token not found in final vocabulary. Padding disabled.")


            logger.info(f"Successfully created WordLevel tokenizer from {source_info} with {len(vocab)} tokens (including added special tokens).")

        except Exception as e:
            logger.error(f"Failed to load or create tokenizer from {source_info}: {e}", exc_info=True) # Log traceback
            # Fallback to a minimal tokenizer ONLY containing UNK
            fallback_vocab = {"<unk>": 0}
            self.tokenizer = Tokenizer(WordLevel(vocab=fallback_vocab, unk_token="<unk>"))
            self.tokenizer.pre_tokenizer = Whitespace()
            # Ensure methods like get_vocab_size don't fail on the fallback
            logger.warning("Using minimal fallback tokenizer with only '<unk>' token.")

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Tokenizes text into a list of token IDs."""
        if not self.tokenizer:
            # This case should ideally not be reached due to fallback, but check anyway
            logger.error("Tokenizer is not initialized (encode). Returning empty list.")
            return []
        try:
            output = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
            return output.ids
        except Exception as e:
            logger.error(f"Error during encoding: {e}", exc_info=True)
            # Fallback: return list containing only UNK token ID if possible
            unk_token_id = self.tokenizer.token_to_id("<unk>")
            if unk_token_id is not None:
                return [unk_token_id]
            else:
                return [0] # Absolute fallback

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Converts a list of token IDs back into text."""
        if not self.tokenizer:
            logger.error("Tokenizer is not initialized (decode). Returning empty string.")
            return ""
        try:
            # Filter out None values which might appear if token_ids contains invalid IDs
            valid_token_ids = [tid for tid in token_ids if tid is not None]
            return self.tokenizer.decode(valid_token_ids, skip_special_tokens=skip_special_tokens)
        except Exception as e:
            logger.error(f"Error during decoding: {e}", exc_info=True)
            return "[DECODING ERROR]"

    def get_vocab_size(self) -> int:
         """Returns the size of the vocabulary."""
         if not self.tokenizer:
             logger.error("Tokenizer is not initialized (get_vocab_size). Returning 0.")
             return 0
         try:
            return self.tokenizer.get_vocab_size()
         except Exception as e:
            logger.error(f"Error getting vocab size: {e}", exc_info=True)
            return 0 # Fallback

    def save(self, file_path: str):
         """Saves the tokenizer state to a file (tokenizer.json format)."""
         if self.tokenizer:
             try:
                 # Ensure directory exists before saving
                 os.makedirs(os.path.dirname(file_path), exist_ok=True)
                 self.tokenizer.save(file_path)
                 logger.info(f"Tokenizer saved to {file_path}")
             except Exception as e:
                 logger.error(f"Failed to save tokenizer to {file_path}: {e}", exc_info=True)
         else:
             logger.error("Cannot save, tokenizer not initialized.")

# Example usage:
# Create a dummy vocab file
# dummy_vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3, "hello": 4, "world": 5}
# with open("dummy_vocab.json", "w") as f:
#     json.dump(dummy_vocab, f)
#
# tokenizer_from_file = HFTokenizer(vocab_file="dummy_vocab.json")
# tokenizer_from_dict = HFTokenizer(vocab_dict=dummy_vocab)
# encoded = tokenizer_from_file.encode("hello world")
# print(f"Encoded: {encoded}") # Example output: Encoded: [1, 4, 5, 2]
# decoded = tokenizer_from_file.decode(encoded)
# print(f"Decoded: {decoded}") # Example output: Decoded: hello world
# print(f"Vocab size: {tokenizer_from_file.get_vocab_size()}")
# tokenizer_from_file.save("saved_tokenizer.json")

