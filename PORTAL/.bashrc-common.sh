
# Firs figure out $LOCAL_BENCH_USER
scriptfile=${BASH_SOURCE[0]}
if [ -z "$scriptfile" ]; then
    >&2 echo "this file should only be sourced"
    exit 1
fi
LOCAL_BENCH_DIR="$(dirname $scriptfile)"
homedir="$(dirname $LOCAL_BENCH_DIR)"
if [ "$(dirname $homedir)" != "/home" ]; then
    >&2 echo "something went terribly wrong with $scriptfile"
    return 1
fi
if [ "$(basename $LOCAL_BENCH_DIR)" = '.bench' ]; then
    LOCAL_BENCH_DIR="$homedir/BENCH"
fi
LOCAL_BENCH_USER="$(basename $homedir)"
if [ -n "$USER" -a "$USER" = "$LOCAL_BENCH_USER" ]; then
    >&2 echo "$scriptfile is not meant to be used by $LOCAL_BENCH_USER"
    return 1
fi
homedir=
scriptfile=

GIT_AUTHOR_NAME="$(git config --global --get 'user.name')"
GIT_AUTHOR_EMAIL="$(git config --global --get 'user.email')"
GIT_COMMITTER_NAME="$GIT_AUTHOR_NAME"
GIT_COMMITTER_EMAIL="$GIT_AUTHOR_EMAIL"


function bench-fix-ssh-agent() {
    if [ -z "$SSH_AUTH_SOCK" ]; then
        >&2 echo "WARNING: no SSH agent running!"
        >&2 echo "(one is required in order to use the bench host)"
        >&2 echo "(be sure to run 'bench-fix-ssh-agent' after starting an SSH agent)"
    else
        echo "fixing permissions on \$SSH_AUTH_SOCK so it can be used by the '$LOCAL_BENCH_USER' user..."

        local agent_dir=$(dirname "$SSH_AUTH_SOCK")
        ( set -x; setfacl -m $LOCAL_BENCH_USER:x $agent_dir; )
        ( set -x; setfacl -m $LOCAL_BENCH_USER:rwx "$SSH_AUTH_SOCK"; )
        echo "...done!"
    fi
}


##################################
# set up for using the bench host

echo
echo '==================================='
echo '=== setting up for benchmarking ==='
echo '==================================='

echo
set -x
alias bench='sudo --login --user $LOCAL_BENCH_USER \
    SUDO_PWD="$(pwd)" \
    SSH_AUTH_SOCK="$SSH_AUTH_SOCK" \
    GIT_AUTHOR_NAME="$GIT_AUTHOR_NAME" \
    GIT_AUTHOR_EMAIL="$GIT_AUTHOR_EMAIL" \
    GIT_COMMITTER_NAME="$GIT_COMMITTER_NAME" \
    GIT_COMMITTER_EMAIL="$GIT_COMMITTER_EMAIL" \
'
# Use of $PWD_INIT assumes the following code in ~$LOCAL_BENCH_USER/.profile:
#  if [ -n "$PWD_INIT" ]; then
#      cd $PWD_INIT
#  fi
alias bench-cwd='bench PWD_INIT="$(pwd)"'
alias bench-git='bench-cwd git'
alias bench-vi='bench-cwd vi -O -u $HOME/.vimrc'
#alias bench-vi='bench-cwd vi -O -u $HOME/.vimrc -i $HOME/.viminfo'
alias bench-view='bench-cwd view -O -u $HOME/.vimrc'

{
portal_config="$LOCAL_BENCH_DIR/portal.json"
BENCH_USER="$(bench jq -r '.send_user' $portal_config)"
BENCH_HOST="$(bench jq -r '.send_host' $portal_config)"
BENCH_PORT="$(bench jq -r '.send_port' $portal_config)"
portal_config=
BENCH_CONN="$BENCH_USER@$BENCH_HOST"
} 2>/dev/null

alias bench-ssh="bench ssh -A -p $BENCH_PORT $BENCH_CONN"
alias bench-scp="bench scp -P $BENCH_PORT"
{ set +x; } 2>/dev/null

echo
bench-fix-ssh-agent

echo
echo "env vars:"
echo
echo "LOCAL_BENCH_USER: $LOCAL_BENCH_USER"
echo "LOCAL_BENCH_DIR:  $LOCAL_BENCH_DIR"
echo "BENCH_USER:       $BENCH_USER"
echo "BENCH_HOST:       $BENCH_HOST"
echo "BENCH_PORT:       $BENCH_PORT"
echo "BENCH_CONN:       $BENCH_CONN"
echo
echo '==================================='
echo '===   done (for benchmarking)   ==='
echo '==================================='
echo
