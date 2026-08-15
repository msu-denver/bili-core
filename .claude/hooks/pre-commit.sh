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
    # Explicit targets, never `.`, mirroring the directory list in
    # run_python_formatters.sh. venv/ sits in the working tree and autoflake does
    # not read .gitignore, so a bare `.` rewrites every installed package in the
    # virtualenv, stripping "unused" imports from the re-export modules that most
    # `__init__.py` files are. Packages keep importing while their public names
    # vanish, so the damage reads as a botched install rather than a formatter.
    black bili > /dev/null 2>&1
    autoflake --recursive --in-place --remove-unused-variables --remove-all-unused-imports bili > /dev/null 2>&1
    isort --profile black bili > /dev/null 2>&1
  fi

  # Stage any auto-fixed files
  git add -u
  exit 0
fi

# Not a git commit, allow
exit 0
