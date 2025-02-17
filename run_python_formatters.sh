#!/bin/bash

# Exit if any command fails, except pylint (we'll handle pylint separately)
set -e

# Turn on command echo for debugging
set -x

# Define the list of directories (space-separated)
directories="bili/"

# Loop through each directory and apply the formatting tools
for dir in $directories; do
    echo "Processing directory: $dir"

    # Run black to format code
    black "$dir"

    # Run autoflake to remove unused variables and imports
    autoflake --in-place --remove-unused-variables --remove-all-unused-imports --recursive "$dir"

    # Run isort to sort imports
    isort --profile=black "$dir"

    echo "Finished processing directory: $dir"
done

# Disable 'set -e' to handle pylint errors manually
set +e

# Run pylint and capture the exit code
pylint_exit_code=0

set +x  # Disable command echo for the output


for dir in $directories; do
    echo "Running pylint for directory: $dir"

    # Run pylint, capture output, and preserve the exit code
    pylint_output=$(pylint "$dir" --msg-template='{msg_id}:{path}:{line},{column}: {msg} ({symbol})' --fail-under=9)
    pylint_status=${PIPESTATUS[0]}  # Capture pylint's exit code

    # Capture the last line which contains the pylint score
    pylint_score=$(echo "$pylint_output" | tail -n 1)

    # Custom sort: path, severity (F > E > W > R > C), message number, then line number numerically
    echo "$pylint_output" \
      | grep -E '^[CREFW]' \
      | sed -e 's/^C/1C/' -e 's/^R/2R/' -e 's/^W/3W/' -e 's/^E/4E/' -e 's/^F/5F/' \
      | sort -t':' -k2,2 -k1,1 -k1.2n -k3,3n \
      | sed -e 's/^1C/C/' -e 's/^2R/R/' -e 's/^3W/W/' -e 's/^4E/E/' -e 's/^5F/F/'

    # Print the pylint score (last line of the original output)
    echo "$pylint_score"

    # Capture the exit code and track if any pylint run fails
    if [ "$pylint_status" -ne 0 ]; then
        pylint_exit_code=1
        echo "Pylint failed for directory: $dir"
    fi
done

set -x  # Re-enable command echo

# Re-enable 'set -e' to catch other errors
set -e

# If pylint failed for any directory, exit with a failure status
if [ $pylint_exit_code -ne 0 ]; then
    echo "Some pylint checks failed"
    exit 1
else
    echo "All pylint checks passed"
    exit 0
fi
