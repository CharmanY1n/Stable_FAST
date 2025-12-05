import logging

import numpy as np
import sentencepiece
from transformers import AutoProcessor

import openpi.shared.download as download


class PaligemmaTokenizer:
    def __init__(self, max_len: int = 48):
        self._max_len = max_len

        path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        with path.open("rb") as f:
            self._tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

    def tokenize(self, prompt: str) -> tuple[np.ndarray, np.ndarray]:
        cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ")
        # tokenize "\n" separately as the "start of answer" token
        tokens = self._tokenizer.encode(cleaned_text, add_bos=True) + self._tokenizer.encode("\n")
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            mask = [True] * tokens_len + padding
            tokens = tokens + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            mask = [True] * self._max_len

        return np.asarray(tokens), np.asarray(mask)


class FASTTokenizer:
    def __init__(self, max_len: int = 256, fast_tokenizer_path: str = "/home/yinxiaoran/data/fast"):
        self._max_len = max_len

        # Download base PaliGemma tokenizer
        paligemma_tokenizer_path = "/home/yinxiaoran/data/openpi_model/openpi_model/big_vision/paligemma_tokenizer.model"
        # path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        # with path.open("rb") as f:
        #     self._paligemma_tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())
        with open(paligemma_tokenizer_path, "rb") as f:
            self._paligemma_tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())
        print(f"=======loaded local paligemma========")

        # Instantiate FAST tokenizer
        self._fast_tokenizer = AutoProcessor.from_pretrained(fast_tokenizer_path, trust_remote_code=True)
        print(f"=======loaded local fast========")
        self._fast_skip_tokens = 128  # Skip last 128 tokens in PaliGemma vocab since they are special tokens

    def tokenize(
        self, prompt: str, state: np.ndarray, actions: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cleaned_text = prompt.lower().strip().replace("_", " ")

        # Convention: state gets discretized into 256 discrete bins (assumed range after normalization: [-1, 1])
        discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        # Convention: prefix includes prompt and string-representation of state, followed by ';'
        state_str = " ".join(map(str, discretized_state))
        prefix = f"Task: {cleaned_text}, State: {state_str};\n"
        prefix_tokens = self._paligemma_tokenizer.encode(prefix, add_bos=True)

        if actions is not None:
            # Tokenize actions with FAST tokenizer --> map to last tokens in PaliGemma vocab
            action_tokens = self._fast_tokenizer(actions[None])[0]
            action_tokens_in_pg = self._act_tokens_to_paligemma_tokens(action_tokens)

            # Convention: postfix contains 'Action:' followed by FAST tokens, followed by '|'
            postfix_tokens = (
                self._paligemma_tokenizer.encode("Action: ")
                + action_tokens_in_pg.tolist()
                + self._paligemma_tokenizer.encode("|", add_eos=True)
            )
        else:
            postfix_tokens = []

        # Create output token sequence & masks
        # AR mask is 0 on prefix (bidirectional attention) and 1 on postfix (causal attention to all previous tokens)
        tokens = prefix_tokens + postfix_tokens
        token_mask = [True] * len(tokens)
        ar_mask = [0] * len(prefix_tokens) + [1] * len(postfix_tokens)
        loss_mask = [False] * len(prefix_tokens) + [True] * len(postfix_tokens)  # Loss on postfix only

        # Pad tokens to max length
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            tokens = tokens + padding
            token_mask = token_mask + padding
            ar_mask = ar_mask + padding
            loss_mask = loss_mask + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            token_mask = token_mask[: self._max_len]
            ar_mask = ar_mask[: self._max_len]
            loss_mask = loss_mask[: self._max_len]

        return np.asarray(tokens), np.asarray(token_mask), np.asarray(ar_mask), np.asarray(loss_mask)

    def extract_actions(self, tokens: np.ndarray, action_horizon: int, action_dim: int) -> np.ndarray:
        # Decode predicted output tokens
        decoded_tokens = self._paligemma_tokenizer.decode(tokens.tolist())
        
        # print("-" * 80)
        # print("!!! DEBUG: Raw decoded string from model !!!")
        # print(decoded_tokens)
        # print("-" * 80)

        # Extract actions from FAST model outputs
        if "Action: " not in decoded_tokens:
            return np.zeros((action_horizon, action_dim), dtype=np.float32)

        # Extract actions from decoded tokens
        raw_action_tokens = np.array(
            self._paligemma_tokenizer.encode(decoded_tokens.split("Action: ")[1].split("|")[0].strip())
        )
        action_tokens = self._act_tokens_to_paligemma_tokens(raw_action_tokens)
        return self._fast_tokenizer.decode(
            [action_tokens.tolist()], time_horizon=action_horizon, action_dim=action_dim
        )[0]

    def _act_tokens_to_paligemma_tokens(self, tokens: np.ndarray | list[int]) -> np.ndarray:
        if isinstance(tokens, list):
            tokens = np.array(tokens)
        return self._paligemma_tokenizer.vocab_size() - 1 - self._fast_skip_tokens - tokens

import etils.epath as epath
import sentencepiece
from transformers import AutoProcessor
from typing_extensions import override
class MyLocalFASTTokenizer(FASTTokenizer):
    """
    Inherits from FASTTokenizer class, reuses the parent class's tokenize and decode logic.
    Overrides the __init__ method to use local fast_tokenizer_path and paligemma_tokenizer_path.
    Overrides the tokenize method to implement custom, more accurate mask calculation.
    """
    @override
    def __init__(self,max_len: int, base_tokenizer_path: str,fast_tokenizer_path: str):
        self._max_len = max_len
        print(f"Loading PaliGemma tokenizer from local: {base_tokenizer_path}")
        path = epath.Path(base_tokenizer_path)
        if not path.exists():
            raise FileNotFoundError(f"The specified local PaliGemma tokenizer file does not exist: {path}")
        with path.open("rb") as f:
            self._paligemma_tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

        print(f"Loading FAST tokenizer from local: {fast_tokenizer_path}")
        self._fast_tokenizer = AutoProcessor.from_pretrained(fast_tokenizer_path, trust_remote_code=True)
        
        self._fast_skip_tokens = 128  # Skip last 128 tokens in PaliGemma vocab since they are special tokens
        
        
class SmolVLMTokenizer:
    """
    Tokenizer designed for the SmolVLM model.
    It directly uses Hugging Face's AutoTokenizer to process text.
    """

    def __init__(self, max_len: int = 256, model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"):
        self._max_len = max_len
        # Load a tokenizer compatible with SmolVLM from Hugging Face Hub
        # trust_remote_code=True is required because SmolVLM uses custom code
        self._tokenizer = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        # SmolVLM uses Llama architecture, usually pad_token is equal to eos_token
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    def tokenize(
        self, prompt: str, state: np.ndarray, actions: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Encode text prompts, states, and actions as model inputs.
        This implementation follows the FASTTokenizer approach, but uses SmolVLM's own tokenizer.
        """
        cleaned_text = prompt.lower().strip().replace("_", " ")

        # state discretization
        discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
        state_str = " ".join(map(str, discretized_state))

        # build prefix text
        prefix_text = f"Task: {cleaned_text}, State: {state_str};\n"

        # If there are actions, build suffix text
        if actions is not None:
            # For SmolVLM, we treat actions as text as well
            action_str = " ".join(map(lambda x: f"{x:.4f}", actions.flatten()))
            full_text = f"{prefix_text}Action: {action_str} |"
        else:
            full_text = prefix_text

        # 
        inputs = self._tokenizer(
            full_text,
            return_tensors="np",
            max_length=self._max_len,
            padding="max_length",
            truncation=True,
        )
        
        tokens = inputs["input_ids"][0]
        token_mask = inputs["attention_mask"][0]

        prefix_tokens = self._tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
        prefix_len = len(prefix_tokens)

        ar_mask = np.zeros(self._max_len, dtype=np.int32)
        loss_mask = np.zeros(self._max_len, dtype=bool)


        if actions is not None:

            action_start_index = prefix_len
            actual_token_len = np.sum(token_mask)
            
            if action_start_index < actual_token_len:
                ar_mask[action_start_index:actual_token_len] = 1
                loss_mask[action_start_index:actual_token_len] = True

        return tokens, token_mask.astype(bool), ar_mask, loss_mask

    def extract_actions(self, tokens: np.ndarray, action_horizon: int, action_dim: int) -> np.ndarray:
        """
        Decode actions from the tokens generated by the model.

        """

        valid_tokens = tokens[tokens != self._tokenizer.pad_token_id]
        decoded_text = self._tokenizer.decode(valid_tokens, skip_special_tokens=True)


        if "Action:" in decoded_text:
            action_part = decoded_text.split("Action:")[1].split("|")[0].strip()
            try:

                action_values = np.fromstring(action_part, dtype=np.float32, sep=' ')
                num_actions = min(len(action_values), action_horizon * action_dim)
                actions = np.zeros(action_horizon * action_dim)
                actions[:num_actions] = action_values[:num_actions]
                return actions.reshape(action_horizon, action_dim)
            except Exception as e:
                logging.warning(f"Failed to decode actions from '{action_part}': {e}")
                return np.zeros((action_horizon, action_dim), dtype=np.float32)
        
        return np.zeros((action_horizon, action_dim), dtype=np.float32)
    