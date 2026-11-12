#!/bin/sh
# Trains the context language model in short, resumable batches, only when the Mac may be used.
#
#   run_batches.sh <total steps> [night start hour=23] [night end hour=7] [day duty=0] [after command]
#
# <after command> runs once, at night, when training has finished (e.g. the GPU pass over the
# typing instances).
#
# Night hours: full speed, 20-minute batches with a 1-minute pause between them.
# Day hours:   <day duty> of the GPU (0.3 = busy 30% of the time; 0 = no daytime work), low priority.
# It waits while on battery, stops when training is complete (exit 0) or fails twice in a row,
# and can be killed at any moment: the checkpoint in runs/lm is at most a few minutes old.
set -u
cd "$(dirname "$0")"
STEPS="${1:?total steps}"
NIGHT_START="${2:-23}"
NIGHT_END="${3:-7}"
DAY_DUTY="${4:-0}"
AFTER="${5:-}"
LOG=runs/lm/batches.log
mkdir -p runs/lm
failures=0

say() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }

is_night() {
    h=$(date +%H | sed 's/^0//'); h=${h:-0}
    if [ "$NIGHT_START" -le "$NIGHT_END" ]; then
        [ "$h" -ge "$NIGHT_START" ] && [ "$h" -lt "$NIGHT_END" ]
    else
        [ "$h" -ge "$NIGHT_START" ] || [ "$h" -lt "$NIGHT_END" ]
    fi
}

say "runner started: $STEPS steps, night $NIGHT_START-$NIGHT_END, day duty $DAY_DUTY"
while :; do
    if ! pmset -g batt | grep -q "AC Power"; then
        sleep 300; continue                      # never train on battery
    fi
    if is_night; then duty=1.0; else duty="$DAY_DUTY"; fi
    if [ "$duty" = "0" ]; then
        sleep 300; continue                      # daytime and no daytime work allowed
    fi
    minutes=20
    if is_night; then                            # never run past the end of the night window
        now=$(( $(date +%H | sed 's/^0//;s/^$/0/') * 60 + $(date +%M | sed 's/^0//;s/^$/0/') ))
        left=$(( (NIGHT_END * 60 - now + 1440) % 1440 ))
        if [ "$left" -lt 4 ]; then sleep 240; continue; fi
        [ "$left" -lt "$minutes" ] && minutes=$left
    fi
    say "batch: duty $duty, $minutes min"
    nice -n 15 .venv/bin/python -I lm.py --train --resume --steps "$STEPS" --minutes "$minutes" --duty "$duty" >> runs/lm/stdout.log 2>&1
    rc=$?
    if [ "$rc" -eq 0 ]; then
        say "training complete"
        [ -z "$AFTER" ] && exit 0
        until is_night && pmset -g batt | grep -q "AC Power"; do sleep 300; done
        say "after-step: $AFTER"
        nice -n 15 sh -c "$AFTER" >> runs/lm/stdout.log 2>&1
        rc=$?
        say "after-step finished with exit code $rc"; exit "$rc"
    elif [ "$rc" -eq 3 ]; then
        failures=0; sleep 60                     # paused at a checkpoint: let the machine breathe
    else
        failures=$((failures + 1))
        say "batch failed with exit code $rc (failure $failures)"
        [ "$failures" -ge 2 ] && exit "$rc"
        sleep 120
    fi
done
