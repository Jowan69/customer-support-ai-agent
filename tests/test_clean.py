from support_agent.text import clean_text


def test_strips_urls_mentions_and_extra_whitespace():
    raw = "@AmazonHelp   my order https://amzn.to/xyz123 hasn't arrived  yet"
    cleaned = clean_text(raw)

    assert "@AmazonHelp" not in cleaned
    assert "https://" not in cleaned
    assert "  " not in cleaned
    assert cleaned == "my order hasn't arrived yet"


def test_quoted_text_is_kept():
    """Customers quote the error they are reporting - it is the informative part."""
    raw = 'the app says "delivery failed" and I don\'t know why'
    cleaned = clean_text(raw)

    assert "delivery failed" in cleaned


def test_emoji_becomes_a_word_instead_of_being_deleted():
    cleaned = clean_text("my order is late \U0001F621")

    assert "\U0001F621" not in cleaned
    assert ":enraged_face:" in cleaned
    assert "my order is late" in cleaned


def test_nfkc_normalises_compatibility_characters():
    assert clean_text("ＯＲＤＥＲ late") == "ORDER late"


def test_collapses_elongation_and_repeated_punctuation():
    assert clean_text("this is sooooo late!!!!!") == "this is soo late!!"


def test_is_idempotent():
    raw = "@AmazonHelp my order is soooo late!!! https://a.co/x \U0001F621"
    once = clean_text(raw)

    assert clean_text(once) == once


def test_non_string_input_is_safe():
    assert clean_text(None) == ""
