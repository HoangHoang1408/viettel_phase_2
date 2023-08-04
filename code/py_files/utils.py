import os


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

prompts = {
    "text_summarization": text_summarization_prompt,
    "conversation_summarization": conversation_summarization_prompt,
    "qa_with_context": qa_with_context_prompt,
    "law_with_context": law_with_context_prompt,
}

from time import perf_counter

# factory decorator to save prompt
import pandas as pd


def log_prompt(model_name, csv_path_to_save="prompt_log.csv"):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = perf_counter()
            prompt = func(*args, **kwargs)
            cur_time = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            total_time = perf_counter() - start
            try:
                df = pd.read_csv(csv_path_to_save)
                if set(df.columns.tolist()) != set(
                    ["model", "generated", "time_taken", "prompt_time"]
                ):
                    raise Exception("Columns are not correct")
            except:
                os.makedirs(os.path.dirname(csv_path_to_save), exist_ok=True)
                df = pd.DataFrame(
                    columns=["model", "generated", "time_taken", "prompt_time"]
                )
            print(f"### Generated in {total_time:.6f} seconds ###\n")
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        [[model_name, prompt, total_time, cur_time]],
                        columns=["model", "generated", "time_taken", "prompt_time"],
                    ),
                ]
            )
            df.drop_duplicates(subset=["generated"], keep="last").to_csv(
                csv_path_to_save, index=False
            )
            return prompt

        return wrapper

    return decorator


def format_prompt(prompt_type, **kwargs):
    if prompt_type not in prompts:
        raise ValueError(
            "Prompt type must be one of the following: {}".format(prompts.keys())
        )
    return prompts[prompt_type].format(**kwargs)


### Need to customize ...
@log_prompt(
    model_name="bloom_sft_3b", csv_path_to_save="../../prompt_logs/prompt_log.csv"
)
def generate(prompt):
    return prompt
