#!/bin/bash

# Main reasoning experiments
# 1. CoT baseline
# 2. CoT with self-consistency
# 3. MAD baseline
# 4. MAD naive (pruned)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# IMPORTANT:Parameters
MODEL_NAME=$1

# Log settings
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/main_reasoning_$(date +%Y%m%d_%H%M%S).log"

failed_scripts=()
start_time=$(date)

scripts=(
    #"cot.sh"
    #"cot_sc.sh"
    "mad.sh"
    "mad_naive.sh"
    "mad_ppl.sh"
)

echo "=========================================="
echo "Starting reasoning scripts"
echo "Model name: $MODEL_NAME"
echo "Start time: $start_time"
echo "Log file: $LOG_FILE"
echo "=========================================="
echo ""

{
    echo "=========================================="
    echo "Main Reasoning Log"
    echo "Model name: $MODEL_NAME"
    echo "Start time: $start_time"
    echo "=========================================="
    echo ""
} >> "$LOG_FILE"

for script in "${scripts[@]}"; do
    script_start=$(date)
    echo "=========================================="
    echo "Current script: $script"
    echo "Start time: $script_start"
    echo "=========================================="
    
    {
        echo "=========================================="
        echo "Script: $script"
        echo "Start time: $script_start"
    } >> "$LOG_FILE"
    
    bash "$SCRIPT_DIR/$script" "$MODEL_NAME"
    exit_code=$?
    script_end=$(date)
    
    if [ "$exit_code" -eq 0 ]; then
        echo "✓ $script completed successfully"
        {
            echo "Status: Success"
            echo "End time: $script_end"
            echo ""
        } >> "$LOG_FILE"
    else
        echo "✗ $script failed, exit code: $exit_code"
        failed_scripts+=("$script|$script_start|$script_end|$exit_code")
        {
            echo "Status: Failed"
            echo "Exit code: $exit_code"
            echo "End time: $script_end"
            echo ""
        } >> "$LOG_FILE"
    fi
    
    echo ""
done

end_time=$(date)

echo "=========================================="
echo "All scripts finished"
echo "End time: $end_time"
echo "=========================================="

if [ ${#failed_scripts[@]} -eq 0 ]; then
    echo ""
    echo "✓ All scripts executed successfully!"
    {
        echo ""
        echo "=========================================="
        echo "Summary"
        echo "End time: $end_time"
        echo "Status: All scripts succeeded"
        echo "=========================================="
    } >> "$LOG_FILE"
else
    echo ""
    echo "✗ Some scripts failed:"
    echo ""
    
    {
        echo ""
        echo "=========================================="
        echo "Summary"
        echo "End time: $end_time"
        echo "Status: ${#failed_scripts[@]} script(s) failed"
        echo "=========================================="
        echo ""
        echo "Failed scripts:"
        echo "----------------------------------------"
    } >> "$LOG_FILE"
    
    for failed in "${failed_scripts[@]}"; do
        IFS='|' read -r script_name start_ts end_ts exit_code <<< "$failed"
        echo "  Script: $script_name"
        echo "  Start: $start_ts"
        echo "  End:   $end_ts"
        echo "  Exit:  $exit_code"
        echo ""
        
        {
            echo "  Script: $script_name"
            echo "  Start: $start_ts"
            echo "  End:   $end_ts"
            echo "  Exit:  $exit_code"
            echo ""
        } >> "$LOG_FILE"
    done
    
    {
        echo "=========================================="
    } >> "$LOG_FILE"
fi

echo "Detailed log saved to: $LOG_FILE"