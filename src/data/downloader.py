from __future__ import annotations

import logging

from src.data.nse_provider import fetch_hdfc_market_data

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    data = fetch_hdfc_market_data(force_refresh=False)
    logger.info("Downloaded %s rows of NSE HDFC Bank data.", len(data))
    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in data.columns]
    logger.info("First date: %s", data["date"].min())
    logger.info("Last date: %s", data["date"].max())
    logger.info("Duplicate dates: %s", int(data["date"].duplicated().sum()))
    logger.info("Missing required fields: %s", missing or "none")


if __name__ == "__main__":
    main()
