"""Word lists for inject_noise(). Every entry maps a clean form to a wrong one, never the reverse."""

# Idioms and phrases people mishear or mistype.
EGGCORNS: dict[str, str] = {
    "for all intents and purposes": "for all intensive purposes",
    "deep-seated": "deep-seeded",
    "moot point": "mute point",
    "peace of mind": "piece of mind",
    "could have": "could of",
    "should have": "should of",
    "would have": "would of",
    "bear in mind": "bare in mind",
    "sneak peek": "sneak peak",
}

# Homophone slips and common misspellings. Two-way pairs (their/there) are fine: edits never overlap,
# so a word is swapped at most once.
HOMOPHONES: dict[str, str] = {
    "their": "there",
    "they're": "their",
    "there": "their",
    "your": "you're",
    "you're": "your",
    "its": "it's",
    "it's": "its",
    "lose": "loose",
    "definitely": "definately",
    "separate": "seperate",
    "receive": "recieve",
    "received": "recieved",
    "weird": "wierd",
    "a lot": "alot",
}

# Keys physically adjacent to each letter on a US QWERTY keyboard.
QWERTY_NEIGHBORS: dict[str, str] = {
    "q": "wa",
    "w": "qase",
    "e": "wsdr",
    "r": "edft",
    "t": "rfgy",
    "y": "tghu",
    "u": "yhji",
    "i": "ujko",
    "o": "iklp",
    "p": "ol",
    "a": "qwsz",
    "s": "awedxz",
    "d": "serfcx",
    "f": "drtgvc",
    "g": "ftyhbv",
    "h": "gyujnb",
    "j": "huikmn",
    "k": "jiolm",
    "l": "kop",
    "z": "asx",
    "x": "zsdc",
    "c": "xdfv",
    "v": "cfgb",
    "b": "vghn",
    "n": "bhjm",
    "m": "njk",
}
