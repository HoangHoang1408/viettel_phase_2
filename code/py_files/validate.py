import json
from copy import deepcopy

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
    pipeline,
)

MODEL_NAME = "general_chatbot_05_08_23"
BASE_MODEL_PATH = "bigscience/bloom-3b"
ADAPTER_PATH = "hoang14/general_chatbot_05_08_23"

ARC_PATH = "/content/arc_validation.json"
ARC_SIZE = 100  # max 2568
MMLU_PATH = "/content/mmlu_validation.json"
MMLU_SIZE = 100  # max 14789

INFERENCE_BATCH_SIZE = 4
INFERENCE_PROMPT = """Cuộc trò chuyện giữa con người và trợ lý AI.
[|Con người|] Lựa chọn một đáp án cho câu hỏi sau đây:
{question}
{choices}
Đáp án:
[|AI|] """

GEN_CONFIG = {
    "temperature": 0.3,
    "top_p": 0.5,
    "top_k": 0,
    "max_new_tokens": 32,
    "repetition_penalty": 1.1,
}


# default generation config
default_gen_config = {
    "temperature": 0.3,
    "top_p": 0.5,
    "top_k": 0,
    "max_new_tokens": 512,
    "repetition_penalty": 1.1,
}


def load_arc(path, size):
    def mapper(x):
        choices = [x[t] for t in ["option_a", "option_b", "option_c", "option_d"]]
        return {
            "input": INFERENCE_PROMPT.format(
                question=x["instruction"],
                choices="\n".join(
                    f"{i}. {c}" for i, c in zip(["A", "B", "C", "D"], choices)
                ),
            ),
            "ref": x["answer"],
            "category": x["id"].split("/")[-1].split("_")[0],
        }

    ds = Dataset.from_json(path)
    ds = (
        ds.shuffle()
        .select(range(min(size, len(ds))))
        .map(mapper, remove_columns=ds.column_names)
    )
    return ds


def load_mmlu(path, size):
    def mapper(x):
        choices = [x[t] for t in ["option_a", "option_b", "option_c", "option_d"]]
        return {
            "input": INFERENCE_PROMPT.format(
                question=x["instruction"],
                choices="\n".join(
                    f"{i}. {c}" for i, c in zip(["A", "B", "C", "D"], choices)
                ),
            ),
            "ref": x["answer"],
            "category": x["id"].split("/")[0],
        }

    ds = Dataset.from_json(path)
    ds = (
        ds.shuffle()
        .select(range(min(size, len(ds))))
        .map(mapper, remove_columns=ds.column_names)
    )
    return ds


# ds format: {'input': prompt_str, 'ref': str[A, B, C, D], 'category': str}
eval_datasets = [
    {
        "name": "ARC",
        "path": ARC_PATH,
        "load_func": load_arc,
        "size": ARC_SIZE,
    },
    {
        "name": "MMLU",
        "path": MMLU_PATH,
        "load_func": load_mmlu,
        "size": MMLU_SIZE,
    },
]


class Inferencer:
    def __init__(
        self,
        adapter_path,
        base_model_path,
        gen_config=default_gen_config,
        tokenizer_max_length=2048,
        human_symbol="[|Con người|]",
    ):
        self.human_symbol = human_symbol
        self.gen_config = deepcopy(gen_config)
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.stopping_criteria = None

        # need to run these functions in order
        print("Loading model and tokenizer...")
        self._load_model_and_tokenizer(
            adapter_path, base_model_path, tokenizer_max_length
        )
        print("Setting stopping criteria...")
        self._set_stopping_criteria([self.human_symbol, self.tokenizer.eos_token])
        print("Building generation pipeline...")
        self._build_pipeline()
        print("Model ready!")

    def _load_model_and_tokenizer(
        self, adapter_path, base_model_name_or_path, tokenizer_max_length
    ):
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                device_map="auto",
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            ),
            trust_remote_code=True,
        )
        self.model = PeftModel.from_pretrained(model, adapter_path)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.model_max_length = tokenizer_max_length

    def _set_stopping_criteria(self, stop_seq_list=[]):
        stop_token_ids_list = [
            torch.tensor(
                self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(x))
            )
            .long()
            .to("cuda")
            for x in stop_seq_list
        ]

        class StopOnTokens(StoppingCriteria):
            def __call__(
                self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
            ) -> bool:
                for stop_ids in stop_token_ids_list:
                    if torch.eq(input_ids[0][-len(stop_ids) :], stop_ids).all():
                        return True
                return False

        self.stopping_criteria = StoppingCriteriaList([StopOnTokens()])

    def _build_pipeline(self):
        self.pipeline = pipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            stopping_criteria=self.stopping_criteria,
            return_full_text=True,
            task="text-generation",
            **self.gen_config,
        )

    def set_gen_config(self, gen_config):
        self.gen_config = gen_config
        self._build_pipeline()

    def generate(self, prompt):
        text_outputs = [x[0]["generated_text"] for x in self.pipeline(prompt)]
        if isinstance(prompt, str):
            text_output = text_outputs[0]
        else:
            text_output = text_outputs
        return text_output


def report(
    generate_func, ds_path_name_load=eval_datasets, bs=INFERENCE_BATCH_SIZE
):  # generate_func: [input_prompts] -> [preds]
    def mapper(x):
        return {
            "input": x["input"],
            "ref": x["ref"],
            "pred": generate_func(x["input"]),
            "category": x["category"],
        }

    def report_for_category(ds):
        res = {x: {"ref": [], "pred": [], "input": []} for x in ds.unique("category")}
        for row in ds:
            res[row["category"]]["ref"].append(row["ref"])
            res[row["category"]]["pred"].append(row["pred"])
            res[row["category"]]["input"].append(row["input"])
        for x in res:
            res[x]["acc"] = sum(
                1 for i, j in zip(res[x]["ref"], res[x]["pred"]) if i == j
            ) / len(res[x]["ref"])
        return res

    datasets = {
        x["name"]: x["load_func"](x["path"], x["size"]) for x in ds_path_name_load
    }
    datasets = {
        x: datasets[x].map(mapper, batched=True, batch_size=bs) for x in datasets
    }
    return {x: report_for_category(datasets[x]) for x in datasets}


inferencer = Inferencer(
    adapter_path=ADAPTER_PATH,
    base_model_path=BASE_MODEL_PATH,
)
inferencer.set_gen_config(GEN_CONFIG)


def generate_func(prompts):
    results = [x[0]["generated_text"] for x in inferencer.pipeline(prompts)]
    indices = [res.find("[|AI|]") for res in results]
    return [res[i + 7 : i + 8] if i != -1 else "" for i, res in zip(indices, results)]


result = report(generate_func)
with open(f"result_{MODEL_NAME}.json", "w") as f:
    json.dump(result, f)
