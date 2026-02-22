#!/bin/bash

# Read hook input from stdin
INPUT=$(cat)

# Extract the command being executed
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // empty')

# Only intercept git commit commands
if [[ "$COMMAND" =~ ^git\ commit ]]; then
  cd "$CLAUDE_PROJECT_DIR" || exit 0

  # Run formatters before allowing the commit
  if [ -f "./run_python_formatters.sh" ]; then
    bash ./run_python_formatters.sh > /dev/null 2>&1
  else
    black . > /dev/null 2>&1
    autoflake --recursive --in-place --remove-unused-variables --remove-all-unused-imports . > /dev/null 2>&1
    isort --profile black . > /dev/null 2>&1
  fi

  # Stage any auto-fixed files
  git add -u
  exit 0
fi

# Not a git commit, allow
exit 0
