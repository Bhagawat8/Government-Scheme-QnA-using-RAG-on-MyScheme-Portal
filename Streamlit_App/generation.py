from langchain_core.language_models import BaseLLM
from langchain_core.outputs import LLMResult, Generation
from typing import List, Tuple, Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_core.pydantic_v1 import PrivateAttr
import torch

# Generation_response
class FalconGenerator(BaseLLM):
    """LangChain-compatible Falcon generator with proper message handling"""
    
    # Configurable parameters
    model_name: str = "tiiuae/Falcon3-3B-Instruct"  # having context lenth of 32k so it is more perfect for RAG pipeline
    max_new_tokens: int = 4096
    temperature: float = 0.2
    repetition_penalty: float = 1.2
    device_map: str = "auto"
    
    # Private attributes
    _model: Any = PrivateAttr()
    _tokenizer: Any = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print(f"Initializing {self.model_name}...")
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map=self.device_map
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def _generate(
        self,
        prompts: List[List[Dict[str, str]]],
        **kwargs: Any,
    ) -> LLMResult:
        generations = []
        for messages in prompts:
            # Convert to chat format
            text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize and move to device
            inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
            
            # Generate tokens
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                repetition_penalty=self.repetition_penalty,
                do_sample=True,
                eos_token_id=self._tokenizer.eos_token_id,
                pad_token_id=self._tokenizer.pad_token_id
            )
            
            # Slice to get only new tokens
            gen_ids = outputs[0][inputs.input_ids.shape[-1]:]
            response = self._tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            
            # Append the response
            generations.append([Generation(text=response)])
        
        return LLMResult(generations=generations)

    @property
    def _llm_type(self) -> str:
        return "falcon-3b-instruct-rag"