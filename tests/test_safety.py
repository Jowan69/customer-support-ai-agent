import pytest

from support_agent.online.safety import check_reply


def test_a_good_reply_passes():
    reply = (
        "I'm sorry your order hasn't arrived. Please check your Orders page for the "
        "latest status, and let us know if it still hasn't turned up in 24-48 hours."
    )

    assert check_reply(reply) == []


def test_prices_dates_and_durations_are_not_violations():
    """False positives here would escalate perfectly good replies."""
    reply = "You were charged £5.99 on 14 October and it should arrive in 24-48 hours."

    assert check_reply(reply) == []


def test_the_link_that_actually_leaked():
    """Verbatim from the first live run: a 2017 t.co link lifted from a retrieved pair."""
    reply = "Check the carrier and tracking: https://t.co/Y5jpI9ys9c. Keep us posted."

    assert check_reply(reply) == ["url"]


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("See t.co/abc123 for details", "url"),
        ("Visit www.amazon.co.uk/orders", "url"),
        ("I can see order 111-5014070-4118645 was dispatched.", "order_number"),
        ("Tracking ID TBA566517950000 shows delivered.", "tracking_id"),
        ("Please write to support@example.com", "email"),
        ("We resent the mail to __email__", "placeholder_leak"),
        ("Reference 123456789012 applies.", "long_number"),
    ],
)
def test_each_leak_is_caught(reply, expected):
    assert expected in check_reply(reply)


def test_several_violations_are_all_reported():
    reply = "Order 111-5014070-4118645 - track at https://t.co/xyz"
    found = check_reply(reply)

    assert "url" in found
    assert "order_number" in found


def test_a_missing_reply_cannot_leak():
    """None is already handled as llm_unavailable; it must not also read as unsafe."""
    assert check_reply(None) == []
    assert check_reply("") == []
