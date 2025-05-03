from dotenv import load_dotenv
import torch
from transformers import (
    BitsAndBytesConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
)
from langchain_huggingface import HuggingFacePipeline

load_dotenv()


class LLModelPipeline:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_skip_modules=None,
            llm_int8_enable_fp32_cpu_offload=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
            quantization_config=self.bnb_config,
            torch_dtype=torch.float16,
        )
        self.generation_pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_token=1024,
            temperature=0.1,
            repetition_penalty=1.1,
        )

    @classmethod
    def load_pipeline(cls):
        config = cls()
        return HuggingFacePipeline(pipeline=config.generation_pipe)
