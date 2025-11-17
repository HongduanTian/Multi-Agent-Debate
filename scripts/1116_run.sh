#!/bin/bash

# Experiments run on 11/16
# 1. Multi-Agent Debate with PPL Prune
# 2. Analysis of Reasoning Paths
# 3. Analysis of Number of Agents
# 4. Analysis of Number of Rounds

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

scripts=(
    #"mad_ppl.sh"
    "analysis_reasonpath.sh"
    "analysis_num_agents.sh"
    "analysis_num_rounds.sh"
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