# BiliCore Security Features

This document describes the multi-tenant security features and cloud-ready architecture implemented in BiliCore.

## Overview

BiliCore implements defense-in-depth security for multi-tenant deployments with:
- **Thread ownership validation** at checkpointer and executor layers
- **Multi-conversation isolation** via conversation_id parameter
- **State-based persistence** for cloud-native deployments
- **On-demand schema migration** with zero downtime
- **Backward compatibility** - all security features are opt-in

## Multi-Tenant Security

### Thread Ownership Validation

All checkpointers validate that thread IDs belong to the authenticated user:

```python
from bili.checkpointers.pg_checkpointer import AsyncPostgresSaver

# Initialize with user_id to enable validation
checkpointer = AsyncPostgresSaver.from_conn_string(
    conn_string="postgresql://user:pass@localhost/bili",
    user_id="user@example.com"
)

# Valid thread IDs
await checkpointer.aget_tuple({"configurable": {"thread_id": "user@example.com"}})
await checkpointer.aget_tuple({"configurable": {"thread_id": "user@example.com_work"}})

# Invalid thread ID - raises PermissionError
await checkpointer.aget_tuple({"configurable": {"thread_id": "other@example.com"}})
# PermissionError: Access denied: thread_id 'other@example.com' does not belong to user 'user@example.com'
```

### Thread ID Pattern

Thread IDs must follow a strict pattern for security:

| Pattern | Description | Example |
|---------|-------------|---------|
| `{user_id}` | Default/single conversation | `user@example.com` |
| `{user_id}_{conversation_id}` | Named conversation | `user@example.com_work` |

**Validation Rules:**
1. Thread ID must exactly match `user_id`, OR
2. Thread ID must start with `{user_id}_` (underscore required)
3. Any other pattern raises `PermissionError`

**Security Examples:**

```python
checkpointer = QueryableMemorySaver(user_id="user@example.com")

# ✅ Valid patterns
checkpointer._validate_thread_ownership("user@example.com")
checkpointer._validate_thread_ownership("user@example.com_work")
checkpointer._validate_thread_ownership("user@example.com_my_project_123")

# ❌ Invalid patterns - raise PermissionError
checkpointer._validate_thread_ownership("other@example.com")
checkpointer._validate_thread_ownership("user@example.com.hacker")  # Wrong separator
checkpointer._validate_thread_ownership("prefix_user@example.com")  # Prefix attack
```

## Multi-Conversation Support

Users can maintain multiple isolated conversation threads simultaneously.

### Basic Usage

```python
from bili.flask_api.flask_utils import handle_agent_prompt

# Default conversation (backward compatible)
response = handle_agent_prompt(user, agent, "Hello")
# Thread ID: user@example.com

# Work conversation
response = handle_agent_prompt(user, agent, "Status update?", conversation_id="work")
# Thread ID: user@example.com_work

# Personal conversation
response = handle_agent_prompt(user, agent, "Vacation ideas?", conversation_id="personal")
# Thread ID: user@example.com_personal
```

### AETHER Integration

```python
from bili.aether.execution.mas_executor import MASExecutor

# Work conversation
work_executor = MASExecutor(
    mas_config=config,
    checkpointer=checkpointer,
    user_id="user@example.com",
    conversation_id="work"
)

# Personal conversation (completely isolated state)
personal_executor = MASExecutor(
    mas_config=config,
    checkpointer=checkpointer,
    user_id="user@example.com",
    conversation_id="personal"
)
```

### Conversation Isolation

Each conversation maintains separate:
- **Message history**: No cross-conversation bleed
- **Checkpoint state**: Independent state snapshots
- **Metadata**: Separate titles, tags, timestamps

```python
# List all conversations for a user
threads = await checkpointer.get_user_threads("user@example.com")

for thread in threads:
    print(f"Conversation: {thread['conversation_id']}")
    print(f"  Thread ID: {thread['thread_id']}")
    print(f"  Messages: {thread['message_count']}")
    print(f"  Last updated: {thread['last_updated']}")
```

## Defense-in-Depth Architecture

Multiple validation layers protect against unauthorized access:

### Layer 1: MASExecutor Validation

```python
class MASExecutor:
    def __init__(self, ..., user_id=None, conversation_id=None):
        # Validates user_id and conversation_id formats
        # Constructs thread_id following security pattern
```

### Layer 2: Checkpointer Validation

```python
class AsyncPostgresSaver:
    async def aput(self, config, checkpoint, metadata, new_versions):
        thread_id = config["configurable"]["thread_id"]
        self._validate_thread_ownership(thread_id)  # ⚠️ Raises PermissionError
        # ... persist checkpoint
```

### Layer 3: Database Isolation

```sql
-- PostgreSQL schema includes indexed user_id column
CREATE TABLE checkpoints (
    thread_id TEXT NOT NULL,
    user_id TEXT,  -- Indexed for fast lookups
    checkpoint BYTEA NOT NULL,
    metadata BYTEA NOT NULL,
    ...
);

CREATE INDEX idx_checkpoints_user_id ON checkpoints(user_id);

-- Queries automatically scoped to user_id
SELECT * FROM checkpoints WHERE user_id = 'user@example.com';
```

## Cloud-Ready State Management

### Before: File-Based Storage (Not Cloud-Ready)

```python
# ❌ File-based state (lost on pod restart)
state_file = f"/data/conversations/{user_id}.jsonl"
with open(state_file, "a") as f:
    f.write(json.dumps(checkpoint) + "\n")
```

**Problems:**
- State lost when Kubernetes pods restart
- Not suitable for multi-instance deployments
- No shared state across replicas
- Manual cleanup required

### After: State-Based Persistence (Cloud-Ready)

```python
# ✅ Database-backed state (survives pod restarts)
checkpointer = AsyncPostgresSaver.from_conn_string(
    conn_string=os.environ["POSTGRES_URL"],
    user_id=user_id
)

# State automatically persisted to database
result = await executor.execute_async({"messages": [msg]})
```

**Benefits:**
- ✅ Survives Kubernetes pod restarts
- ✅ Shared state across multiple pod replicas
- ✅ Automatic state recovery on initialization
- ✅ Database-managed retention and cleanup
- ✅ Multi-instance safe with connection pooling

## On-Demand Schema Migration

Database schema changes occur only when `user_id` is first provided.

### PostgreSQL Migration

```python
# First use with user_id triggers migration
checkpointer = AsyncPostgresSaver.from_conn_string(
    conn_string="postgresql://...",
    user_id="user@example.com"  # ⚠️ Triggers migration check
)
```

**Migration Steps:**
1. Check if `user_id` column exists
2. If missing, add column: `ALTER TABLE checkpoints ADD COLUMN user_id TEXT`
3. Create index: `CREATE INDEX idx_checkpoints_user_id ON checkpoints(user_id)`
4. Continue normal operations

**Zero Downtime:**
- Migration runs automatically during initialization
- No manual intervention required
- Existing data remains accessible
- No service interruption

### MongoDB Migration

```python
# First use with user_id triggers schema update
checkpointer = AsyncMongoDBSaver.from_conn_string(
    conn_string="mongodb://...",
    user_id="user@example.com"  # ⚠️ Triggers schema update
)
```

**Migration Steps:**
1. Add `user_id` field to new documents
2. Create index: `db.checkpoints.createIndex({"user_id": 1})`
3. Existing documents work without `user_id` (backward compatible)

## Backward Compatibility

All security features are **opt-in** via the `user_id` parameter.

### Without user_id (Backward Compatible)

```python
# No user_id - validation disabled
checkpointer = QueryableMemorySaver()

# Any thread ID accepted
checkpointer.put({"configurable": {"thread_id": "any_thread"}}, checkpoint, metadata, {})
checkpointer.put({"configurable": {"thread_id": "another_thread"}}, checkpoint, metadata, {})
```

### With user_id (Multi-Tenant Security Enabled)

```python
# With user_id - validation enabled
checkpointer = QueryableMemorySaver(user_id="user@example.com")

# Only user's threads accepted
checkpointer.put({"configurable": {"thread_id": "user@example.com"}}, checkpoint, metadata, {})
checkpointer.put({"configurable": {"thread_id": "user@example.com_work"}}, checkpoint, metadata, {})

# Other users' threads rejected
checkpointer.put({"configurable": {"thread_id": "other@example.com"}}, checkpoint, metadata, {})
# PermissionError: Access denied
```

## Testing Multi-Tenant Security

### Test Ownership Validation

```python
import pytest
from bili.checkpointers.memory_checkpointer import QueryableMemorySaver

def test_thread_ownership_validation():
    checkpointer = QueryableMemorySaver(user_id="user@example.com")

    # Valid threads
    valid_threads = [
        "user@example.com",
        "user@example.com_work",
        "user@example.com_conv_123"
    ]

    for thread_id in valid_threads:
        checkpointer._validate_thread_ownership(thread_id)  # Should not raise

    # Invalid threads
    invalid_threads = [
        "other@example.com",
        "malicious@example.com",
        "user@example.com.hacker"
    ]

    for thread_id in invalid_threads:
        with pytest.raises(PermissionError, match="Access denied"):
            checkpointer._validate_thread_ownership(thread_id)
```

### Test Conversation Isolation

```python
def test_conversation_isolation():
    checkpointer = QueryableMemorySaver(user_id="user@example.com")

    # Create work conversation
    config_work = {"configurable": {"thread_id": "user@example.com_work"}}
    checkpointer.put(config_work, checkpoint_work, metadata, {})

    # Create personal conversation
    config_personal = {"configurable": {"thread_id": "user@example.com_personal"}}
    checkpointer.put(config_personal, checkpoint_personal, metadata, {})

    # Verify isolation
    work_state = checkpointer.get_tuple(config_work)
    personal_state = checkpointer.get_tuple(config_personal)

    assert work_state != personal_state
    assert work_state.config["configurable"]["thread_id"] == "user@example.com_work"
    assert personal_state.config["configurable"]["thread_id"] == "user@example.com_personal"
```

## Flask API Integration

### Multi-Conversation Routes

```python
from flask import Flask, request, g, jsonify
from bili.flask_api.flask_utils import handle_agent_prompt

@app.route("/chat", methods=["POST"])
@auth_required(AUTH_MANAGER, required_roles=["user"])
def chat():
    """Chat endpoint with multi-conversation support."""
    data = request.get_json()

    # Extract conversation_id (optional)
    prompt = data.get("prompt", "")
    conversation_id = data.get("conversation_id")  # None for default conversation

    # g.user populated by @auth_required decorator
    return handle_agent_prompt(g.user, conversation_agent, prompt, conversation_id)
```

### Example Requests

```bash
# Default conversation
curl -X POST http://localhost:5000/chat \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello"}'

# Work conversation
curl -X POST http://localhost:5000/chat \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Status update?", "conversation_id": "work"}'

# Personal conversation
curl -X POST http://localhost:5000/chat \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Vacation ideas?", "conversation_id": "personal"}'
```

## Best Practices

### 1. Always Use user_id in Production

```python
# ❌ Don't do this in production
checkpointer = AsyncPostgresSaver.from_conn_string(conn_string)  # No user_id

# ✅ Do this instead
checkpointer = AsyncPostgresSaver.from_conn_string(
    conn_string,
    user_id=authenticated_user_email
)
```

### 2. Validate Thread IDs at Application Boundaries

```python
# Validate before passing to executor
def create_executor(user_email, conversation_id):
    if conversation_id and not conversation_id.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Invalid conversation_id format")

    return MASExecutor(
        mas_config=config,
        checkpointer=checkpointer,
        user_id=user_email,
        conversation_id=conversation_id
    )
```

### 3. Use Conversation IDs for Logical Separation

```python
# Separate by purpose
conversation_ids = {
    "work": "Work-related conversations",
    "personal": "Personal topics",
    "research": "Research projects",
    "support": "Customer support tickets"
}
```

### 4. List User Conversations Before Creating New Ones

```python
# Check existing conversations
threads = await checkpointer.get_user_threads(user_email)
existing_ids = {t["conversation_id"] for t in threads}

# Create only if needed
if "work" not in existing_ids:
    executor = MASExecutor(..., conversation_id="work")
```

## Security Checklist

- [ ] All production checkpointers initialized with `user_id`
- [ ] Thread IDs validated at executor and checkpointer layers
- [ ] Conversation IDs sanitized at application boundaries
- [ ] Database indexes created on `user_id` column
- [ ] Tests cover ownership validation edge cases
- [ ] Flask routes extract `conversation_id` from requests
- [ ] Error messages don't leak thread IDs of other users
- [ ] Monitoring in place for `PermissionError` exceptions

## See Also

- [ARCHITECTURE.md](./ARCHITECTURE.md) - Overall architecture documentation
- [LANGGRAPH.md](./LANGGRAPH.md) - LangGraph workflow documentation
- [bili/aether/README.md](../bili/aether/README.md) - AETHER framework documentation
- [bili/checkpointers/README.md](../bili/checkpointers/README.md) - Checkpointer implementation details
