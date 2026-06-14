"""
CLI commands for dataset management.

Usage:
    python -m gently.dataset.cli aggregate [--full]
    python -m gently.dataset.cli stats
    python -m gently.dataset.cli serve [--port PORT]
"""

import argparse
import logging
import sys
from pathlib import Path

from .aggregator import DatasetAggregator
from .schema import DEFAULT_DB_PATH, get_connection, get_database_stats

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def cmd_aggregate(args):
    """Run data aggregation."""
    db_path = Path(args.db) if args.db else DEFAULT_DB_PATH

    logger.info(f"Starting aggregation to {db_path}")
    logger.info(f"Mode: {'full' if args.full else 'incremental'}")

    aggregator = DatasetAggregator(db_path=db_path)

    try:
        stats = aggregator.aggregate_all(incremental=not args.full)

        print("\n=== Aggregation Complete ===")
        print(f"Sessions:     {stats['sessions']}")
        print(f"Embryos:      {stats['embryos']}")
        print(f"Volumes:      {stats['volumes']}")
        print(f"Images:       {stats['images']}")
        print(f"Ground Truth: {stats['ground_truth']}")
        print(f"\nDatabase: {db_path}")

    finally:
        aggregator.close()


def cmd_stats(args):
    """Show database statistics."""
    db_path = Path(args.db) if args.db else DEFAULT_DB_PATH

    if not db_path.exists():
        logger.error("Database not found: %s", db_path)
        logger.error("Run 'python -m gently.dataset.cli aggregate' first.")
        sys.exit(1)

    conn = get_connection(db_path)
    stats = get_database_stats(conn)
    conn.close()

    print("\n=== Dataset Statistics ===")
    print(f"Database: {db_path}")
    print(f"Size: {db_path.stat().st_size / 1024 / 1024:.2f} MB")
    print()
    print("Record Counts:")
    print(f"  Sessions:        {stats.get('sessions', 0):,}")
    print(f"  Embryos:         {stats.get('embryos', 0):,}")
    print(f"  Volumes:         {stats.get('volumes', 0):,}")
    print(f"  Images:          {stats.get('images', 0):,}")
    print(f"  Ground Truth:    {stats.get('ground_truth', 0):,}")
    print(f"  Perception Runs: {stats.get('perception_runs', 0):,}")
    print(f"  Predictions:     {stats.get('predictions', 0):,}")
    print()
    print(f"Unique Embryo-Sessions: {stats.get('unique_embryo_sessions', 0):,}")
    print()
    if stats.get("earliest_volume"):
        print(f"Date Range: {stats['earliest_volume'][:10]} to {stats['latest_volume'][:10]}")


def cmd_serve(args):
    """Start the web explorer server."""
    db_path = Path(args.db) if args.db else DEFAULT_DB_PATH

    if not db_path.exists():
        logger.error("Database not found: %s", db_path)
        logger.error("Run 'python -m gently.dataset.cli aggregate' first.")
        sys.exit(1)

    try:
        from .explorer_server import DatasetExplorer

        explorer = DatasetExplorer(db_path=db_path, port=args.port)
        explorer.run()
    except ImportError as e:
        logger.error("Error importing explorer: %s", e)
        logger.error("Make sure FastAPI is installed: pip install fastapi uvicorn")
        sys.exit(1)


def cmd_query(args):
    """Run a SQL query on the database."""
    db_path = Path(args.db) if args.db else DEFAULT_DB_PATH

    if not db_path.exists():
        logger.error("Database not found: %s", db_path)
        sys.exit(1)

    conn = get_connection(db_path)

    try:
        cursor = conn.execute(args.sql)
        rows = cursor.fetchall()

        if not rows:
            print("No results")
            return

        # Print header
        columns = [desc[0] for desc in cursor.description]
        print("\t".join(columns))
        print("-" * 80)

        # Print rows
        for row in rows[: args.limit]:
            print("\t".join(str(v) if v is not None else "NULL" for v in row))

        if len(rows) > args.limit:
            print(f"\n... and {len(rows) - args.limit} more rows")

    finally:
        conn.close()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Embryo Dataset Management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run incremental aggregation
    python -m gently.dataset.cli aggregate

    # Run full aggregation (re-scan everything)
    python -m gently.dataset.cli aggregate --full

    # Show database statistics
    python -m gently.dataset.cli stats

    # Start web explorer
    python -m gently.dataset.cli serve --port 8765

    # Run a SQL query
    python -m gently.dataset.cli query "SELECT * FROM sessions LIMIT 5"
        """,
    )

    parser.add_argument("--db", help=f"Database path (default: {DEFAULT_DB_PATH})")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Aggregate command
    agg_parser = subparsers.add_parser("aggregate", help="Aggregate data into database")
    agg_parser.add_argument(
        "--full", action="store_true", help="Run full aggregation (not incremental)"
    )

    # Stats command
    subparsers.add_parser("stats", help="Show database statistics")

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start web explorer")
    serve_parser.add_argument(
        "--port", type=int, default=8765, help="Port to serve on (default: 8765)"
    )

    # Query command
    query_parser = subparsers.add_parser("query", help="Run SQL query")
    query_parser.add_argument("sql", help="SQL query to run")
    query_parser.add_argument(
        "--limit", type=int, default=100, help="Max rows to display (default: 100)"
    )

    args = parser.parse_args()

    if args.command == "aggregate":
        cmd_aggregate(args)
    elif args.command == "stats":
        cmd_stats(args)
    elif args.command == "serve":
        cmd_serve(args)
    elif args.command == "query":
        cmd_query(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
