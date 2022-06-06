#!/usr/bin/env bash

# See PORTAL/bot-cronjobs.sh

# First ensure the SSH agent is set up.
unset SSH_AUTH_SOCK
source $HOME/.ssh/agent.sh &>/dev/null
if [ -z "$SSH_AUTH_SOCK" -o ! -e "$SSH_AUTH_SOCK" ]; then
    >&2 echo 'ERROR: SSH agent not running'
    exit 1
elif ! ssh-add -L | grep 'id_rsa_github' >/dev/null; then
    >&2 echo 'ERROR: GitHub SSH key missing'
    exit 1
fi

if [ -z "$USER" ]; then
    # It must be a cron job.
    export USER=$LOGNAME
fi

# Then set up env vars for the upload.
export GIT_AUTHOR_NAME='bot'
export GIT_AUTHOR_EMAIL='faster.cpython@gmail.com'
export GIT_COMMITTER_NAME=$GIT_AUTHOR_NAME
export GIT_COMMITTER_EMAIL=$GIT_AUTHOR_EMAIL

# Finally, run the benchmarks (and upload).
jobs_py=$(dirname $(realpath $0))/../scripts/jobs.py
(
set -x
$jobs_py add compile-bench \
    --benchmarks='-flaskblogging' \
    --user bot \
    --upload \
    main
)
# --upload-arg '<no-push>'
# --fake-success
