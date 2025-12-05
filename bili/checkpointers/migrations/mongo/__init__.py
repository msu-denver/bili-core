"""
MongoDB-specific checkpointer migrations.

These migrations handle format changes specific to MongoDB storage,
particularly the JsonPlusSerializer changes from LangGraph upgrades.
"""

from bili.checkpointers.migrations.mongo.v1_to_v2 import migrate_v1_to_v2

__all__ = ["migrate_v1_to_v2"]
