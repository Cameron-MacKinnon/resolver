import re

import requests

from .cache_config import USER_AGENT


class RulesLinkNotFoundError(Exception):
    """raised when the rules page's .txt download link can't be found"""


class WotcClient:
    """Handles all contact with Wizards of the Coast's official Comprehensive Rules page."""

    def __init__(self) -> None:
        self.rules_page_url = "https://magic.wizards.com/en/rules"
        self.headers = {"User-Agent": USER_AGENT}

    def _discover_latest_txt_url(self) -> str:
        """Scrape the rules page for the current Comprehensive Rules .txt download link.

        The download filename is versioned by date and changes with every rules
        update, so it can't be hardcoded; instead we retrieve it from this (hopefully)
        stable page: "https://magic.wizards.com/en/rules"
        """
        # get page contents
        response = requests.get(self.rules_page_url, headers=self.headers)
        response.raise_for_status()

        # search for link text ending in ".txt"
        match = re.search(
            r'href="(https://media\.wizards\.com[^"]+?\.txt)"', response.text
        )
        if match is None:
            raise RulesLinkNotFoundError(
                f'no .txt download link found on "{self.rules_page_url}"'
            )
        return match.group(1)

    def fetch_comprehensive_rules(self) -> str:
        """Discover and fetch the current Comprehensive Rules text"""
        txt_url = self._discover_latest_txt_url()
        response = requests.get(txt_url, headers=self.headers)
        response.raise_for_status()

        # the server sends Content-Type: text/plain with no charset, so requests
        # falls back to ISO-8859-1 by default even though the body is UTF-8 -
        # override explicitly to avoid mangling curly quotes/dashes
        response.encoding = "utf-8"
        return response.text
