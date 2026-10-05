from incorrecter import SYSTEM_PROMPT, TASK_INSTRUCTION


def test_prompts_keep_the_chats_wording():
    assert SYSTEM_PROMPT == (
        "You are Incorrecter. Your job is to subtly corrupt pristine text with realistic human typos, "
        "eggcorns, and minor grammar lapses."
    )
    assert TASK_INSTRUCTION == (
        "Sabotage this text naturally. Inject subtle typos, wrong homophones, or malapropisms to make it "
        "look authentically human-typed."
    )
