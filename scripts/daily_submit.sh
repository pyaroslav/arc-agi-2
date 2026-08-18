#!/bin/bash
# Standing daily draw-max submitter (through Nov 1, 2026).
# Each day at ~00:20 UTC: if today's submission slot is unused, submit the
# current best-known kernel (read fresh from outputs/daily_submit_config each
# day, format: "<kernel-ref> <version> <message...>"). Manual/experiment
# submissions earlier in the day take precedence via the slot-used guard.
# Stop: kill this PID (logged) or delete outputs/daily_submit_config.
COMP=arc-prize-2026-arc-agi-2
CONF=outputs/daily_submit_config
cd "$(dirname "$0")/.." || exit 1

while [ "$(date -u +%Y%m%d)" -le "20261101" ]; do
  # sleep until next 00:20 UTC
  now=$(date -u +%s)
  tomorrow=$(date -u -d "tomorrow 00:20" +%s)
  sleep $(( tomorrow - now ))
  [ -f "$CONF" ] || { echo "$(date -u) no config; stopping"; break; }
  today=$(date -u +%Y-%m-%d)
  last=$(kaggle competitions submissions -c $COMP 2>/dev/null | sed -n 3p | awk '{print $3}')
  if [ "$last" = "$today" ]; then
    echo "$(date -u) slot already used today ($last); skip"
    continue
  fi
  read -r KERNEL VERSION MSG < "$CONF"
  echo "$(date -u) submitting $KERNEL v$VERSION"
  kaggle competitions submit -c $COMP -k "$KERNEL" -v "$VERSION" -f submission.json \
    -m "${MSG:-draw-max resubmit} [auto $(date -u +%m%d)]" 2>&1
  sleep 60
  kaggle competitions submissions -c $COMP 2>/dev/null | sed -n 3p
done
echo "$(date -u) daily submitter finished"
