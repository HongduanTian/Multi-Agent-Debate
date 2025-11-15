#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

scripts=(
    "cot.sh"
    "cot_sc.sh"
    "mad.sh"
    "mad_naive.sh"
    "mad_ppl.sh"
)

for script in "${scripts[@]}"; do
    echo "=========================================="
    echo "Current script: $script"
    echo "Current time: $(date)"
    echo "=========================================="
    
    bash "$SCRIPT_DIR/$script"
    
    if [ $? -eq 0 ]; then
        echo "✓ $script completed"
    else
        echo "✗ $script failed, exit code: $?"
        # exit 1
    fi
    
    echo ""
done

echo "=========================================="
echo "All scripts completed"
echo "End time: $(date)"
echo "=========================================="