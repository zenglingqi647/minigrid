"""
Chat with a model with command line interface.

Usage:
python3 -m fastchat.serve.cli --model lmsys/vicuna-7b-v1.5
python3 -m fastchat.serve.cli --model lmsys/fastchat-t5-3b-v1.0
"""
import argparse
import os

from fastchat.model.model_adapter import add_model_args
from fastchat.modules.awq import AWQConfig
from fastchat.modules.exllama import ExllamaConfig
from fastchat.modules.xfastertransformer import XftConfig
from fastchat.modules.gptq import GptqConfig
from fastchat.utils import str_to_torch_dtype
'''inference.py'''
"""Inference for FastChat models."""
import abc
from typing import Optional

from fastchat.conversation import get_conv_template
from fastchat.model.model_adapter import (
    load_model,
    get_conversation_template,
    get_generate_stream_function,
)
from fastchat.modules.awq import AWQConfig
from fastchat.modules.gptq import GptqConfig
from fastchat.modules.exllama import ExllamaConfig
from fastchat.modules.xfastertransformer import XftConfig
from fastchat.utils import get_context_length


class ChatIO(abc.ABC):

    @abc.abstractmethod
    def prompt_for_input(self, prompt: str) -> str:
        """Prompt for input from a role."""

    @abc.abstractmethod
    def prompt_for_output(self, role: str):
        """Prompt for output from a role."""

    @abc.abstractmethod
    def stream_output(self, output_stream):
        """Stream output."""

    @abc.abstractmethod
    def print_output(self, text: str):
        """Print output."""


class SimpleChatIO(ChatIO):

    def __init__(self, multiline: bool = False):
        self._multiline = multiline

    def prompt_for_input(self, prompt: str) -> str:
        """role input prompt"""
        if not self._multiline:
            return prompt

        prompt_data = []
        line = prompt
        while True:
            prompt_data.append(line.strip())
            try:
                line = input()
            except EOFError as e:
                break
        return "\n".join(prompt_data)

    def prompt_for_output(self) -> None:
        """indicates the role of output. Skipped."""
        pass

    def stream_output(self, output_stream):
        pre = 0
        txt = ""
        for outputs in output_stream:
            output_text = outputs["text"]
            output_text = output_text.strip().split(" ")
            now = len(output_text) - 1
            if now > pre:
                txt += " ".join(output_text[pre:now])
                pre = now
        txt += " ".join(output_text[pre:])
        return txt

    def print_output(self, text: str):
        print(text)


def chat_loop(
    model,
    tokenizer,
    model_path: str,
    device: str,
    conv_template: Optional[str],
    conv_system_msg: Optional[str],
    temperature: float,
    repetition_penalty: float,
    max_new_tokens: int,
    chatio: ChatIO,
    judge_sent_end: bool = True,
    history: bool = True,
    prompt: str = "",
):
    # Model
    generate_stream_func = get_generate_stream_function(model, model_path)

    model_type = str(type(model)).lower()
    is_t5 = "t5" in model_type

    # Hardcode T5's default repetition penalty to be 1.2
    if is_t5 and repetition_penalty == 1.0:
        repetition_penalty = 1.2

    # Set context length
    context_len = get_context_length(model.config)

    # Chat
    def new_chat():
        if conv_template:
            conv = get_conv_template(conv_template)
        else:
            conv = get_conversation_template(model_path)
        if conv_system_msg is not None:
            conv.set_system_message(conv_system_msg)
        return conv

    conv = None

    while True:
        if not history or not conv:
            conv = new_chat()

        try:
            inp = chatio.prompt_for_input(prompt)
        except EOFError:
            inp = ""

        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        gen_params = {
            "model": model_path,
            "prompt": prompt,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
            "max_new_tokens": max_new_tokens,
            "stop": conv.stop_str,
            "stop_token_ids": conv.stop_token_ids,
            "echo": False,
        }

        try:
            chatio.prompt_for_output(conv.roles[1])
            output_stream = generate_stream_func(
                model,
                tokenizer,
                gen_params,
                device,
                context_len=context_len,
                judge_sent_end=judge_sent_end,
            )
            outputs = chatio.stream_output(output_stream)
            conv.update_last_message(outputs.strip())

        except KeyboardInterrupt:
            print("stopped generation.")
            # If generation didn't finish
            if conv.messages[-1][1] is None:
                conv.messages.pop()
                # Remove last user message, so there isn't a double up
                if conv.messages[-1][0] == conv.roles[0]:
                    conv.messages.pop()


def setup(args):
    if args.gpus:
        if len(args.gpus.split(",")) < args.num_gpus:
            raise ValueError(f"Larger --num-gpus ({args.num_gpus}) than --gpus {args.gpus}!")
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        os.environ["XPU_VISIBLE_DEVICES"] = args.gpus
    if args.enable_exllama:
        exllama_config = ExllamaConfig(
            max_seq_len=args.exllama_max_seq_len,
            gpu_split=args.exllama_gpu_split,
        )
    else:
        exllama_config = None
    if args.enable_xft:
        xft_config = XftConfig(
            max_seq_len=args.xft_max_seq_len,
            data_type=args.xft_dtype,
        )
        if args.device != "cpu":
            print("xFasterTransformer now is only support CPUs. Reset device to CPU")
            args.device = "cpu"
    else:
        xft_config = None
    try:
        model, tokenizer = load_model(
            args.model_path,
            device=args.device,
            num_gpus=args.num_gpus,
            max_gpu_memory=args.max_gpu_memory,
            dtype=str_to_torch_dtype(args.dtype),
            load_8bit=args.load_8bit,
            cpu_offloading=args.cpu_offloading,
            gptq_config=GptqConfig(
                ckpt=args.gptq_ckpt or args.model_path,
                wbits=args.gptq_wbits,
                groupsize=args.gptq_groupsize,
                act_order=args.gptq_act_order,
            ),
            awq_config=AWQConfig(
                ckpt=args.awq_ckpt or args.model_path,
                wbits=args.awq_wbits,
                groupsize=args.awq_groupsize,
            ),
            exllama_config=exllama_config,
            xft_config=xft_config,
            revision=args.revision,
        )
        return model, tokenizer
    except KeyboardInterrupt:
        print("exit...")


def query(args, model, tokenizer, prompt, chatio):
    chat_loop(
        model,
        tokenizer,
        args.model_path,
        args.device,
        args.conv_template,
        args.conv_system_msg,
        args.temperature,
        args.repetition_penalty,
        args.max_new_tokens,
        chatio,
        judge_sent_end=args.judge_sent_end,
        history=not args.no_history,
        prompt=prompt,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_model_args(parser)
    parser.add_argument("--conv-template", type=str, default=None, help="Conversation prompt template.")
    parser.add_argument("--conv-system-msg", type=str, default=None, help="Conversation system message.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--no-history", action="store_true")
    parser.add_argument(
        "--style",
        type=str,
        default="simple",
        help="Display style.",
    )
    parser.add_argument(
        "--multiline",
        action="store_true",
        help="Enable multiline input. Use ESC+Enter for newline.",
    )
    parser.add_argument(
        "--judge-sent-end",
        action="store_true",
        help="Whether enable the correction logic that interrupts the output of sentences due to EOS.",
    )
    args = parser.parse_args()

    prompt = "hello world"
    chatio = SimpleChatIO(args.multiline)
    model, tokenizer = setup(args)
    query(args, model, tokenizer, prompt, chatio)