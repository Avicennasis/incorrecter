"""The memorization check: does a model continue a training seed verbatim? (Task 14.)

`build_prompt` continues an email from its first words with no system role, exactly the
prompt format the operational run uses; `hit_run` reports whether the continuation contains a
verbatim `run`-word stretch of the seed text AFTER the prefix, so a continuation that merely
echoes the prefix never counts.
"""


def build_prompt(prefix_words: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": f"Continue this email:\n\n{prefix_words}"}]


def hit_run(continuation: str, seed_text: str, prefix: str, run: int = 8) -> bool:
    """True when the continuation contains a verbatim `run`-word stretch of the seed text after the
    prefix words. Whitespace-immune: the runner joins the prefix with single spaces, which must not
    hide a seed's line breaks from the match."""
    pfx = [w.casefold() for w in prefix.split()]
    seed_words = seed_text.split()
    n = len(pfx)
    for i in range(len(seed_words) - n + 1):
        if [w.casefold() for w in seed_words[i : i + n]] == pfx:
            window = seed_words[i + n :]
            if len(window) < run:
                return False
            runs = {" ".join(window[j : j + run]).casefold() for j in range(len(window) - run + 1)}
            words = continuation.split()
            return any(" ".join(words[j : j + run]).casefold() in runs for j in range(max(len(words) - run + 1, 0)))
    return False
