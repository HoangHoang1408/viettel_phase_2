import os
from copy import deepcopy
from time import perf_counter

import pandas as pd
import torch
from peft import PeftModel
from traitlets import Bool
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
    pipeline,
)


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
    + "[|Con người|] Trả lời câu hỏi pháp luật dựa vào những điều luật liên quan dưới đây. Chỉ được trả lời dựa trên thông tin nằm trong điều luật được cung cấp.\nCâu hỏi: {question}\nĐiều luật liên quan:\n{context}\n[|AI|] "
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
        log=True,
        csv_path_to_save_logs="./prompt_log.csv",
        log_model_name="bloom_sft_3b",
    ):
        start = perf_counter()
        text_output = self.pipeline(prompt)[0]["generated_text"]
        total_time = perf_counter() - start
        if log:
            self._log(log_model_name, text_output, total_time, csv_path_to_save_logs)
        print(f"### Generated in {total_time:.6f} seconds ###\n")
        return text_output
