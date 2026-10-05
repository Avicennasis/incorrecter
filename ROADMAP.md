# Roadmap

Open items, in rough priority order.

## 1. A near-miss check for the meaning judge

The meaning judge calibrates (58/58 on unchanged pairs, 58/58 on mismatched pairs),
and the model keeps meaning on 0.96 of held-out texts at the recommended settings.
What the calibration cannot show is how often the judge misses a *subtle* change.
Add a calibration set of pairs with exactly one meaning-changing edit (a dropped
"not", a changed number, my → your) and require "no".

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
