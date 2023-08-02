import random
from pprint import pprint
from time import sleep

import gradio as gr
import torch
from peft import PeftConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
    pipeline,
)

default_gen_config = {
    "temperature": 0.3,
    "top_p": 0.5,
    "top_k": 0,
    "max_new_tokens": 512,
    "repetition_penalty": 1.1,
}


class ChatBot:
    def __init__(
        self,
        apdapter_path,
        gen_config=default_gen_config,
    ):
        self.sleep_time = 0.02
        self.gen_config = gen_config
        self._set_stopping_criteria()
        self._load_llm(apdapter_path)
        self.set_gen_config(self.gen_config)
        self.system_message = "Đoạn hội thoại giữa con người và AI"
        self.history = []  # [(human_input, bot_response)]
        self.human_symbol = "[|Con người|]"
        self.ai_symbol = "[|AI|]"

    def _load_llm(self, adapter_path):
        peft_config = PeftConfig.from_pretrained(adapter_path)
        model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
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
        self.tokenizer = AutoTokenizer.from_pretrained(
            peft_config.base_model_name_or_path
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.model_max_length = 512

    def _set_stopping_criteria(self):
        stop_seq_list = [self.human_symbol, self.tokenizer.eos_token]
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

    def _construct_input_conversation(self, human_input):
        conversation = self.get_conversation_sofar()
        return (
            conversation + f"{self.human_symbol} {human_input}\n" + f"{self.ai_symbol} "
        )

    def _clear_history(self):
        self.history = []

    def set_gen_config(self, gen_config):
        self.llm = pipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            stopping_criteria=self.stopping_criteria,
            return_full_text=True,
            task="text-generation",
            **gen_config,
        )

    def get_conversation_sofar(self):
        conversation = self.system_message + "\n"
        for human_input, ai_response in self.history:
            conversation += (
                f"{self.human_symbol} {human_input}\n"
                + f"{self.ai_symbol} {ai_response}\n"
            )
        return conversation

    def generate(self, human_input):
        prompt = self._construct_input_conversation(human_input)
        gen_conversation = self.llm(prompt)[0]["generated_text"]
        gen_conversation = gen_conversation.removesuffix(self.human_symbol)
        ai_response = gen_conversation.split(self.ai_symbol)[-1].strip()
        self.history.append((human_input, ai_response))
        return ai_response

    def render(self):
        def chat(human_input, history=self.history):
            yield "", history + [(human_input, None)]
            response = ""
            for letter in self.generate(human_input):
                sleep(self.sleep_time)
                response += letter
                yield "", history + [(human_input, response)]
        
        with gr.Blocks() as demo:
            gr.Markdown("## Chat bot demo")
            with gr.Tabs():
                with gr.TabItem("Chat"):
                    chatbot = gr.Chatbot(height=600)
                    message = gr.Textbox(placeholder="Type your message here...")
                    message.submit(chat, [message, chatbot], [message, chatbot])
                with gr.TabItem("Settings"):
                    gr.Slider(minimum=0, maximum=1, step=0.01, label="Confidence")
                    with gr.Row():
                        setting_btn = gr.Button("Save settings")
                        reset_setting_btn = gr.Button("Reset settings")

        demo.queue().launch(share=True)
