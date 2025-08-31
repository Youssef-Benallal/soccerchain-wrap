# soccerchain_wrap/cli.py
import logging


def configure_logging() -> None:
    """Configure global logging settings for the soccerchain_wrap package."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler()]
    )


def main() -> None:
    """Main entry point for the soccerchain_wrap CLI."""
    configure_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting soccerchain_wrap pipeline...")

    # TODO: Add CLI argument parsing or pipeline triggers here


if __name__ == "__main__":
    main()