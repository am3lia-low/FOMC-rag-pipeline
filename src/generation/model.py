import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from src.config import GENERATION_MODEL


def load_mistral():
    print(f"Loading {GENERATION_MODEL} with 4-bit quantization...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        GENERATION_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    print(f"Model loaded. Device: {model.device} | "
          f"Memory: {model.get_memory_footprint() / 1e9:.1f} GB")

    return model, tokenizer


def generate_answer(prompt, model, tokenizer, max_new_tokens=512, temperature=0.3):
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=4096
    ).to(model.device)

    prompt_length = inputs['input_ids'].shape[1]
    start_time = time.time()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    gen_time = time.time() - start_time
    tokens_generated = outputs[0].shape[0] - prompt_length

    answer = tokenizer.decode(
        outputs[0][prompt_length:], skip_special_tokens=True
    ).strip()

    print(f"  [{tokens_generated} tokens in {gen_time:.1f}s "
          f"({tokens_generated/gen_time:.0f} tok/s) | prompt: {prompt_length} tokens]")

    return answer
