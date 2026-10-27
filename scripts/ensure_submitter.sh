#!/bin/bash
# @reboot guard: start the daily draw-max submitter if it is not running.
export PATH="/home/yaroub/anaconda3/bin:$PATH"
cd /home/yaroub/Python/BUISINESS/KinetiStackLLC/Projects/KAGGLE/ARC/ARC-AGI-2 || exit 1
pgrep -f "^/bin/bash scripts/daily_submit[.]sh" >/dev/null && exit 0
setsid nohup scripts/daily_submit.sh >> outputs/daily_submit.log 2>&1 < /dev/null &
