from time import sleep

import gradio as gr
from inferencer import Inferencer, PromptType
from memory_constructor import (
    BaseMemoryConstrucor,
    DynamicWindowLengthMemoryConstructor,
    FixedWindowLengthMemoryConstructor,
    FullMemoryConstructor,
)
from retriever import Retriever


class ChatBot:
    def __init__(
        self,
        inferencer: Inferencer,
        retriever: Retriever,
        memory: BaseMemoryConstrucor = FixedWindowLengthMemoryConstructor(
            system_message="Đoạn hội thoại giữa con người và AI.\n",
            human_symbol="[|Con người|]",
            ai_symbol="[|AI|]",
        ),
    ):
        self.sleep_time = 0.02
        self.inferencer = inferencer
        self.memory = memory
        self.retriever = retriever

    def remove_suffix(self, text):
        return (
            text.removesuffix(self.memory.human_symbol)
            .split(self.memory.ai_symbol)[-1]
            .strip()
        )

    def free_style_answer(self, human_input):
        prompt = self.memory.construct_input_memory(human_input)
        gen_conversation = self.inferencer.generate(prompt)
        ai_response = self.remove_suffix(gen_conversation)
        self.memory.add_to_memory(human_input, ai_response)
        return ai_response

    def answer(self, human_input, retrieve=True):
        if retrieve:
            if len(self.memory.memory) > 0:
                full_conversation = (
                    self.memory.get_used_memory() + f"\n[|Con người|] {human_input}"
                )
                con_sum_prompt = self.inferencer.format_prompt(
                    PromptType.CONVERSATION_SUMMARIZATION,
                    conversation=full_conversation,
                )
                self.inferencer.summarize_mode(True)
                retrieve_input = self.inferencer.generate(con_sum_prompt)
                self.inferencer.summarize_mode(False)
            else:
                retrieve_input = human_input

            print(retrieve_input)
            retrieved_contexts = self.retriever.retrieve(
                retrieve_input, only_semantic=True
            )
            context_prompt = self.inferencer.format_prompt(
                PromptType.LAW_WITH_CONTEXT,
                question=human_input,
                contexts="\n".join(retrieved_contexts),
            )
            law_answer = self.inferencer.generate(context_prompt)
            
            print(law_answer)
            law_answer = self.remove_suffix(law_answer)
            self.memory.add_to_memory(human_input, law_answer)
            return law_answer
        else:
            return self.free_style_answer(human_input)

    def render(self, share=True):
        def chat(human_input, history=self.memory.memory):
            yield "", history + [(human_input, None)]
            response = ""
            for letter in self.answer(human_input):
                sleep(self.sleep_time)
                response += letter
                yield "", history + [(human_input, response)]

        def clear_conversation():
            self.memory.clear_memory()
            return self.memory.memory

        def pop_message():
            self.memory.pop_from_memory()
            return self.memory.memory

        with gr.Blocks() as demo:
            gr.Markdown("## Chat bot demo")
            with gr.Tabs():
                with gr.TabItem("Chat"):
                    chatbot = gr.Chatbot(height=600)
                    message = gr.Textbox(placeholder="Type your message here...")
                    with gr.Row():
                        # buttons with green background
                        clear_btn = gr.Button("Clear chat history", type="secondary")
                        pop_btn = gr.Button("Pop last message", type="secondary")
                    clear_btn.click(clear_conversation, [], [chatbot])
                    pop_btn.click(pop_message, [], [chatbot])
                    message.submit(chat, [message, chatbot], [message, chatbot])
                with gr.TabItem("Settings"):
                    gr.Slider(minimum=0, maximum=1, step=0.01, label="Confidence")
                    with gr.Row():
                        setting_btn = gr.Button("Save settings")
                        reset_setting_btn = gr.Button("Reset settings")

        demo.queue().launch(share=share)
