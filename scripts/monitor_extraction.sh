#!/bin/bash
# Monitor AUDETER extraction progress
while true; do
    count=$(grep -c "^Processing:" logs/audeter_real_extraction.log 2>/dev/null || echo "0")
    errors=$(grep -c "Error processing" logs/audeter_real_extraction.log 2>/dev/null || echo "0")
    total=19784
    pct=$((count * 100 / total))
    echo "$(date '+%H:%M:%S') - Processed: $count/$total ($pct%) - Errors: $errors"
    
    # Check if process is still running
    if ! pgrep -f "extract.*audeter_real" > /dev/null; then
        echo "Extraction complete!"
        break
    fi
    sleep 60
done
