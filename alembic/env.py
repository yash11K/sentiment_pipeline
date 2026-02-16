from logging.config import fileConfig
import os
import sys
from pathlib import Path

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from storage.models import Base, get_database_url
from utils.logger import get_logger

logger = get_logger(__name__)

# this is the Alembic Config object
config = context.config

# Load .env so APP_ENV and DATABASE_URL_* are available
from dotenv import load_dotenv
load_dotenv()

# Resolve database URL using the same logic as the app
database_url = get_database_url()
env_label = os.getenv('APP_ENV', 'dev').lower()
logger.database(f"Alembic targeting {env_label} database")
config.set_main_option('sqlalchemy.url', database_url)

# Interpret the config file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Model metadata for autogenerate
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    logger.database("Running migrations in offline mode")
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()
    logger.success("Offline migrations complete")


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    logger.database("Running migrations in online mode")
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, 
            target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()
    logger.success("Online migrations complete")


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
