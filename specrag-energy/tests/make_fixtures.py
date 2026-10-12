"""Tiny random models and fake data for the smoke test (no download needed).

    python tests/make_fixtures.py
"""
import json
import os
import random

from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
from transformers import (PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM,
                          RobertaConfig, RobertaForMaskedLM)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
os.makedirs(OUT, exist_ok=True)
rng = random.Random(0)
WORDS = ("the of and in to was is for on as by with he at from his an were are which this "
         "true false health claim vaccine study paris london river king film song year city "
         "rationale response evidence question answer yes no explanation supports").split()

corpus = [" ".join(rng.choice(WORDS) for _ in range(30)) for _ in range(2000)]
corpus += ["## Rationale: x ## Response: y", "Do you think the explanation supports the answers? (Yes or No)"] * 50


def make_tok(specials, chat=False):
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(corpus, trainers.BpeTrainer(
        vocab_size=600, special_tokens=["<unk>"] + specials,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    return tok


# tiny Qwen2 model
t = make_tok(["<|endoftext|>", "<|im_start|>", "<|im_end|>"])
lm_tok = PreTrainedTokenizerFast(tokenizer_object=t, eos_token="<|im_end|>", pad_token="<|endoftext|>",
                                 unk_token="<unk>")
lm_tok.chat_template = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n"
                        "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
for name, layers in (("tiny-drafter", 2), ("tiny-verifier", 3)):
    cfg = Qwen2Config(vocab_size=len(lm_tok), hidden_size=64, intermediate_size=128, num_hidden_layers=layers,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
                      eos_token_id=lm_tok.eos_token_id, pad_token_id=lm_tok.pad_token_id,
                      bos_token_id=None, tie_word_embeddings=True)
    model = Qwen2ForCausalLM(cfg)
    model.generation_config.eos_token_id = [lm_tok.eos_token_id]
    model.generation_config.pad_token_id = lm_tok.pad_token_id
    model.generation_config.repetition_penalty = 1.05   # as in Qwen2.5-Instruct
    model.save_pretrained(os.path.join(OUT, name))
    lm_tok.save_pretrained(os.path.join(OUT, name))

# tiny RoBERTa model
t = make_tok(["<s>", "</s>", "<pad>", "<mask>"])
mlm_tok = PreTrainedTokenizerFast(tokenizer_object=t, bos_token="<s>", eos_token="</s>", sep_token="</s>",
                                  cls_token="<s>", pad_token="<pad>", mask_token="<mask>", unk_token="<unk>")
from tokenizers.processors import RobertaProcessing  # noqa: E402
mlm_tok._tokenizer.post_processor = RobertaProcessing(("</s>", mlm_tok.sep_token_id), ("<s>", mlm_tok.cls_token_id))
cfg = RobertaConfig(vocab_size=len(mlm_tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                    num_attention_heads=4, max_position_embeddings=520, pad_token_id=mlm_tok.pad_token_id,
                    type_vocab_size=1)
RobertaForMaskedLM(cfg).save_pretrained(os.path.join(OUT, "tiny-inbedder"))
mlm_tok.save_pretrained(os.path.join(OUT, "tiny-inbedder"))

# fake data files in Self-RAG format
raw = os.path.join(OUT, "raw")
os.makedirs(raw, exist_ok=True)


def ctxs():
    # some unscored passages at ranks 6-10, like the real TriviaQA file
    scored = [{"id": str(i), "title": rng.choice(WORDS).title(), "score": str(round(2.0 - 0.05 * i, 3)),
               "text": " ".join(rng.choice(WORDS) for _ in range(60))} for i in range(15)]
    extra = [{"title": "NOSCORE", "text": "inserted passage without score"} for _ in range(rng.randint(0, 5))]
    return scored[:5] + extra + scored[5:]


with open(os.path.join(raw, "health_claims_processed.jsonl"), "w") as fh:
    for i in range(60):
        fh.write(json.dumps({"id": f"ph{i}", "question": f"claim {i} " + " ".join(rng.sample(WORDS, 8)),
                             "answers": [rng.choice(["true", "false"])], "ctxs": ctxs()}) + "\n")
with open(os.path.join(raw, "triviaqa_test_w_gs.jsonl"), "w") as fh:
    for i in range(60):
        fh.write(json.dumps({"id": f"tq{i}", "question": f"question {i} " + " ".join(rng.sample(WORDS, 8)),
                             "answers": [rng.choice(WORDS), rng.choice(WORDS)], "ctxs": ctxs()}) + "\n")

cfg = json.load(open(os.path.join(os.path.dirname(OUT), "..", "config.json")))
cfg["models"].update({"drafter": os.path.join(OUT, "tiny-drafter"), "verifier": os.path.join(OUT, "tiny-verifier"),
                      "embedder": os.path.join(OUT, "tiny-inbedder"), "dtype": "float32",
                      "attn_implementation": "sdpa"})
cfg["generation"]["max_new_tokens"] = 24
cfg["data"].update({"pubhealth_raw": os.path.join(raw, "health_claims_processed.jsonl"),
                    "triviaqa_raw": os.path.join(raw, "triviaqa_test_w_gs.jsonl"),
                    "prepared_dir": os.path.join(OUT, "prepared")})
cfg["experiment"].update({"cooldown_ms": 0, "warmup_queries": 2, "thermal_gate_enabled": False})
cfg["server_port"] = 8799
json.dump(cfg, open(os.path.join(OUT, "config_smoke.json"), "w"), indent=2)
print("fixtures written to", OUT)
