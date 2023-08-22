import json
import os
from copy import deepcopy
from time import perf_counter

import pandas as pd
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

BASE_MODEL_PATH = "bigscience/bloom-3b"
ADAPTER_PATH = "hoang14/general_chatbot_05_08_23"

ARC_PATH = "/content/arc_validation.json"
ARC_SIZE = 100
MMLU_PATH = "/content/mmlu_validation.json"
MMLU_SIZE = 100

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


# prompt templates
class PromptType:
    TEXT_SUMMARIZATION = "text_summarization"
    CONVERSATION_SUMMARIZATION = "conversation_summarization"
    QA_WITH_CONTEXT = "qa_with_context"
    LAW_WITH_CONTEXT = "law_with_context"


conversation_system_prompt = "Cuộc trò chuyện giữa con người và trợ lý AI.\n"

conversation_summarization_prompt = "Tóm tắt ngắn gọn đoạn hội thoại sau đây:\n{conversation}\nĐoạn hội thoại đã được tóm tắt:\n"
text_summarization_prompt = (
    conversation_system_prompt
    + "[|Con người|] Tóm tắt ngắn gọn đoạn văn bản sau đây:\n{context}\n[|AI|] "
)
qa_with_context_prompt = (
    conversation_system_prompt
    + "[|Con người|] Trả lời câu hỏi dựa vào đoạn văn bản dưới đây. Chỉ được trả lời dựa trên thông tin nằm trong văn bản được cung cấp.\nCâu hỏi: {question}\nĐoạn văn bản: {context}\n[|AI|] "
)
law_with_context_prompt = (
    conversation_system_prompt
    + "[|Con người|] Trả lời câu hỏi pháp luật dựa vào những điều luật liên quan dưới đây. Chỉ được trả lời dựa trên thông tin nằm trong điều luật được cung cấp.\nCâu hỏi: {question}\nĐiều luật liên quan:\n{context}\n[|AI|] Câu trả lời tóm tắt: "
)

# default generation config
default_gen_config = {
    "temperature": 0.3,
    "top_p": 0.5,
    "top_k": 0,
    "max_new_tokens": 512,
    "repetition_penalty": 1.1,
}


class Inferencer:
    def __init__(
        self,
        adapter_path,
        base_model_path,
        gen_config=default_gen_config,
        tokenizer_max_length=2048,
        human_symbol="[|Con người|]",
    ):
        self.prompt_templates = {
            PromptType.TEXT_SUMMARIZATION: text_summarization_prompt,
            PromptType.CONVERSATION_SUMMARIZATION: conversation_summarization_prompt,
            PromptType.QA_WITH_CONTEXT: qa_with_context_prompt,
            PromptType.LAW_WITH_CONTEXT: law_with_context_prompt,
        }
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

    def _log(
        self, model_name, generated, time_taken, csv_path_to_save_logs="prompt_log.csv"
    ):
        try:
            df = pd.read_csv(csv_path_to_save_logs)
            if set(df.columns.tolist()) != set(
                ["model", "generated", "time_taken", "prompt_time"]
            ):
                raise Exception("Columns are not correct")
        except:
            os.makedirs(os.path.dirname(csv_path_to_save_logs), exist_ok=True)
            df = pd.DataFrame(
                columns=["model", "generated", "time_taken", "prompt_time"]
            )
        cur_time = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    [[model_name, generated, time_taken, cur_time]],
                    columns=["model", "generated", "time_taken", "prompt_time"],
                ),
            ]
        )
        df.drop_duplicates(subset=["generated"], keep="last").to_csv(
            csv_path_to_save_logs, index=False
        )

    def summarize_mode(self, state: bool):
        if state == True:
            self._set_stopping_criteria([self.tokenizer.eos_token])
        else:
            self._set_stopping_criteria([self.tokenizer.eos_token, self.human_symbol])
        self._build_pipeline()

    def set_gen_config(self, gen_config):
        self.gen_config = gen_config
        self._build_pipeline()

    def format_prompt(self, prompt_type, **kwargs):
        if prompt_type not in self.prompt_templates:
            raise ValueError(
                "Prompt type must be one of the following: {}".format(
                    self.prompt_templates.keys()
                )
            )
        return self.prompt_templates[prompt_type].format(**kwargs)

    def generate(
        self,
        prompt,
        log=False,
        csv_path_to_save_logs="./prompt_log.csv",
        log_model_name="bloom_sft_3b",
    ):
        start = perf_counter()
        text_outputs = [x[0]["generated_text"] for x in self.pipeline(prompt)]
        if isinstance(prompt, str):
            text_output = text_outputs[0]
        else:
            text_output = text_outputs
        total_time = perf_counter() - start
        if log:
            self._log(log_model_name, text_output, total_time, csv_path_to_save_logs)
            print(f"### Generated in {total_time:.6f} seconds ###\n")
        return text_output


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
temp = [
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


def report(
    generate_func, ds_path_name_load=temp, bs=INFERENCE_BATCH_SIZE
):  # generate_func: [input] -> [pred]
    def mapper(x):
        return {
            "input": x["input"],
            "ref": x["ref"],
            "pred": generate_func(x["input"]),
            "category": x["category"],
        }

    def report_for_category(ds):
        res = {x: {"ref": [], "pred": []} for x in ds.unique("category")}
        for row in ds:
            res[row["category"]]["ref"].append(row["ref"])
            res[row["category"]]["pred"].append(row["pred"])
        for x in res:
            res[x]["acc"] = sum(
                1 for i, j in zip(res[x]["ref"], res[x]["pred"]) if i == j
            ) / len(res[x]["ref"])
        return res

    datasets = {
        x["name"]: x["load_func"](x["path"], x["size"]).map(
            mapper, batched=True, batch_size=bs
        )
        for x in ds_path_name_load
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
    return [res[i + 7 : i + 8].strip() for i, res in zip(indices, results)]


result = report(generate_func)

# check if file exist if not create and then write json
with open(f"result_{ADAPTER_PATH.split('/')[-1]}.json", "w") as f:
    json.dump(result, f)
