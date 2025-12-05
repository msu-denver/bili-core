"""
Checkpointer migrations package.

This package contains migration functions for upgrading checkpoint data
between different format versions.

Migrations are organized by checkpointer type:
- mongo/: MongoDB-specific migrations (JsonPlusSerializer format changes)
- pg/: PostgreSQL-specific migrations (future use)

Migrations are registered automatically when their package is imported.
Each checkpointer should import its own migration package.
"""

# Note: Checkpointer-specific migrations are imported by each checkpointer
# to avoid unnecessary dependencies. See:
# - bili.checkpointers.migrations.mongo for MongoDB migrations
# - bili.checkpointers.migrations.pg for PostgreSQL migrations (future)
