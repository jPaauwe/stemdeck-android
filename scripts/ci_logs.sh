#!/bin/bash
# Publishes this job's *.log files (and any extra paths given) to the branch
# ci-logs/<job>, and repeats the tail of each log as an annotation when the job
# failed, so a failure can be read without access to the Actions log archive.
#   scripts/ci_logs.sh <job> <run id> <sha> <failed: true|false> [extra paths...]
set -u
job=$1 run=$2 sha=$3 failed=$4
shift 4
dir=$(mktemp -d)
cp -f ./*.log "$dir"/ 2>/dev/null || true
for p in "$@"; do cp -rf "$p" "$dir"/ 2>/dev/null || true; done
if [ "$failed" = true ]; then
  for f in ./*.log; do
    [ -f "$f" ] || continue
    body=$(tail -n 40 "$f" | cut -c1-300 | sed 's/%/%25/g' | awk '{printf "%s%%0A", $0}')
    echo "::error title=tail of ${f#./}::$body"
  done
fi
cd "$dir" || exit 0
echo "run $run job $job sha $sha" > INFO.txt
git init -q -b logs && git add -A
git -c user.name=ci -c user.email=ci@localhost commit -q -m "logs $run $job"
for i in 1 2 3; do
  if out=$(git push -f "https://x-access-token:${GH_TOKEN}@github.com/${GITHUB_REPOSITORY}" "logs:refs/heads/ci-logs/$job" 2>&1); then
    exit 0
  fi
  sleep 5
done
echo "::warning title=ci-logs push failed::$(echo "$out" | tail -n 5 | sed "s/${GH_TOKEN}/***/g" | awk '{printf "%s%%0A", $0}')"
exit 0
