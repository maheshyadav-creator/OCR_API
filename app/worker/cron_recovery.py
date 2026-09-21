from app.database.connection import (
    close_connection_pool,
    initialize_connection_pool,
)
from app.worker.recovery import recover_pending_messages
from app.worker.redis_consumer import create_consumer_group


def main() -> None:
    print("=" * 60)
    print("Cron recovery job started.")
    print("=" * 60)

    initialize_connection_pool()

    try:
        create_consumer_group()
        recovered_count = recover_pending_messages()

        print(
            f"Cron recovery job finished. "
            f"Recovered jobs: {recovered_count}"
        )

    except Exception as exc:
        print(f"Cron recovery job failed: {exc}")
        raise

    finally:
        close_connection_pool()
        print("Cron recovery job finished.")
        print("=" * 60)


if __name__ == "__main__":
    main()