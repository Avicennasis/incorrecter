# Roadmap

Open items, in rough priority order.

## 1. A meaning-preservation judge

Iterate the evaluation prompt or switch models until the meaning judge calibrates
(>= 0.98 agreement on unchanged pairs, >= 0.95 on changed pairs), then measure
meaning preservation on the held-out set.

## 2. A pipe-through CLI

`pbpaste | incorrecter | pbcopy` — a tiny CLI around the model, plus a rule-only
mode that uses the noise engine directly with no model.

## 3. Base-model comparison

Run the eval against current small models (Qwen3-0.6B, gemma-3-270m, LFM2.5-1.2B,
SmolLM3-3B) to check Qwen2.5-0.5B-Instruct is still the right base.

## 4. Compare against existing tools

Compare `nlpaug`'s `KeyboardAug`/`SpellingAug` with the hand-rolled noise engine;
adopt if better. Also evaluate unsloth's mlx trainer against plain mlx-lm.

## 5. Grow the word lists

More eggcorns and homophones (then/than, affect/effect, "should of"), regional
dialect slips, work-chat slang. Every addition maps a clean form to a wrong one and
keeps `tests/test_lexicon.py` green.
