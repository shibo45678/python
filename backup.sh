#!/bin/bash
cd "$(dirname "$0")"
git add .
git commit -m "backup $(date)" 2>/dev/null
git push origin main