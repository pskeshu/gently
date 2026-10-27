"""Who a request is, on a rig with no accounts: the name given at the landing."""

from types import SimpleNamespace

from gently.ui.web import accounts
from gently.ui.web.auth import OPERATOR_COOKIE, current_username


def _request(**cookies):
    return SimpleNamespace(cookies=cookies, client=None)


def test_the_operator_cookie_names_the_person_when_there_are_no_accounts(monkeypatch):
    monkeypatch.setattr(accounts, "_store", None)
    assert current_username(_request()) is None
    assert current_username(_request(**{OPERATOR_COOKIE: "  pskeshu "})) == "pskeshu"
    assert current_username(_request(**{OPERATOR_COOKIE: "x" * 80})) == "x" * 40
