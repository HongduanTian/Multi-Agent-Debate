#!/bin/bash

# Experiments run
# 1. Multi-Agent Debate with PPL Prune
# 2. Analysis of Reasoning Paths
# 3. Analysis of Number of Agents
# 4. Analysis of Number of Rounds

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Create log directory and file
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/analysis_exp_$(date +%Y%m%d_%H%M%S).log"

# Track failed scripts
failed_scripts=()
start_time=$(date)

scripts=(
    "mad_ppl.sh"
    "analysis_reasonpath.sh"
    "analysis_num_agents.sh"
    "analysis_num_rounds.sh"
)

echo "=========================================="
echo "Starting experiment scripts execution"
echo "Start time: $start_time"
echo "Log file: $LOG_FILE"
echo "=========================================="
echo ""

# Write start information to log
{
    echo "=========================================="
    echo "Experiment Execution Log"
    echo "Start time: $start_time"
    echo "=========================================="
    echo ""
} >> "$LOG_FILE"

for script in "${scripts[@]}"; do
    script_start_time=$(date)
    echo "=========================================="
    echo "Current script: $script"
    echo "Start time: $script_start_time"
    echo "=========================================="
    
    # Write script information to log
    {
        echo "=========================================="
        echo "Script: $script"
        echo "Start time: $script_start_time"
    } >> "$LOG_FILE"
    
    bash "$SCRIPT_DIR/$script"
    exit_code=$?
    
    script_end_time=$(date)
    
    if [ "$exit_code" -eq 0 ]; then
        echo "✓ $script completed successfully"
        {
            echo "Status: Success"
            echo "End time: $script_end_time"
            echo ""
        } >> "$LOG_FILE"
    else
        echo "✗ $script failed, exit code: $exit_code"
        # Record failed script information
        failed_scripts+=("$script|$script_start_time|$script_end_time|$exit_code")
        {
            echo "Status: Failed"
            echo "Exit code: $exit_code"
            echo "End time: $script_end_time"
            echo ""
        } >> "$LOG_FILE"
    fi
    
    echo ""
done

end_time=$(date)

# Print summary information
echo "=========================================="
echo "All scripts execution completed"
echo "End time: $end_time"
echo "=========================================="

# Handle failed scripts
if [ ${#failed_scripts[@]} -eq 0 ]; then
    echo ""
    echo "✓ All scripts executed successfully!"
    {
        echo ""
        echo "=========================================="
        echo "Execution Summary"
        echo "End time: $end_time"
        echo "Status: All scripts executed successfully"
        echo "=========================================="
    } >> "$LOG_FILE"
else
    echo ""
    echo "✗ The following scripts failed:"
    echo ""
    
    # Print failed scripts and write to log
    {
        echo ""
        echo "=========================================="
        echo "Execution Summary"
        echo "End time: $end_time"
        echo "Status: ${#failed_scripts[@]} script(s) failed"
        echo "=========================================="
        echo ""
        echo "Failed scripts list:"
        echo "----------------------------------------"
    } >> "$LOG_FILE"
    
    for failed_info in "${failed_scripts[@]}"; do
        IFS='|' read -r script_name failed_start failed_end exit_code <<< "$failed_info"
        echo "  Script: $script_name"
        echo "  Start time: $failed_start"
        echo "  End time: $failed_end"
        echo "  Exit code: $exit_code"
        echo ""
        
        {
            echo "  Script: $script_name"
            echo "  Start time: $failed_start"
            echo "  End time: $failed_end"
            echo "  Exit code: $exit_code"
            echo ""
        } >> "$LOG_FILE"
    done
    
    {
        echo "=========================================="
    } >> "$LOG_FILE"
fi

echo "Detailed log saved to: $LOG_FILE"