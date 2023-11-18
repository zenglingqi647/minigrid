# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import List, Optional

import fire

from llama import Llama, Dialog


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
):
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 512.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 8.
        max_gen_len (int, optional): The maximum length of generated sequences. If None, it will be
            set to the model's max sequence length. Defaults to None.
    """
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    dialogs: List[Dialog] = [
        [
            # {
            #     "role": "system",
            #     "content": "Always answer with emojis",
            # },
            {
                "role":
                    "user",
                "content":
                    """You are an agent in a Minigrid environment. Your mission is go to a green key in as few steps aspossible.
Your agent's direction is currently right.
Your agent can only see in front of itself. It cannot see blocked objects. Here is the vision of youagent. Assume your agent is at the center of the last row. The first row is the furthest from yourfront:
unseen, unseen, unseen, unseen, unseen, unseen, unseenunseen, unseen, unseen, unseen, unseen, unseen, unseenunseen, wall, wall, wall, wall, wall, wall.
unseen, yellow closed door, empty, empty, empty, empty, empty,unseen, wall, empty, empty, empty, empty, empty,unseen, wall, empty, empty, empty, grey box, emptyunseen, wall, empty, empty, empty, empty, empty.
You have a set of skills at your disposal. They are listed in the following:
Skill 1: Go to Obiect (in the same room)
Skill 2: Open door (in the same room)
Skill 3: Pickup an item (in the same room)
Skill 4. Put an item next to an item (in the same room)
Skill 5. Unlock a door (in the same room)
Skill 6: Find an object (in a random room)
Skill 7: Go to the green object (in a random n).
Generate a probability vector for using each of the skills given the circumstance, in a comma:
             """
            },
        ],
    ]
    results = generator.chat_completion(
        dialogs,  # type: ignore
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    for dialog, result in zip(dialogs, results):
        for msg in dialog:
            print(f"{msg['role'].capitalize()}: {msg['content']}\n")
        print(f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}")
        print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)
