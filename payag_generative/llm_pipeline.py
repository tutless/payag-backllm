import os
from dotenv import load_dotenv
import torch
from transformers import (
    BitsAndBytesConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
)
from huggingface_hub import login
from langchain_huggingface import HuggingFacePipeline

load_dotenv()
login(os.getenv("HF_TOKEN"))


class LLModelPipeline:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_skip_modules=None,
            llm_int8_enable_fp32_cpu_offload=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, token=os.getenv("HF_TOKEN")
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
            token=os.getenv("HF_TOKEN"),
            quantization_config=self.bnb_config,
            torch_dtype=torch.float16,
        )
        self.generation_pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=1024,
            temperature=0.1,
            repetition_penalty=1.1,
        )

    @classmethod
    def load_pipeline(cls, model_id):
        config = cls(model_id=model_id)
        return HuggingFacePipeline(pipeline=config.generation_pipe)
