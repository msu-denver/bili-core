"""
PostgreSQL-specific checkpointer migrations.

This package is reserved for future PostgreSQL-specific migrations.
Currently, no migrations are registered as PostgreSQL checkpointing
format remains compatible.

Note: PostgreSQL uses msgpack for some blob fields, so migrations
may differ from MongoDB's JSON-based format migrations.
"""

# No migrations registered yet - PostgreSQL format is currently compatible
# Future migrations would be registered here like:
# from bili.checkpointers.migrations.pg.v1_to_v2 import migrate_v1_to_v2
