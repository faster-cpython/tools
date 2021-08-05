#!/bin/env bash

SCRIPTS_DIR=$(dirname $BASH_SOURCE)
if [ -z "$SCRIPTS_DIR" ]; then
    SCRIPTS_DIR=$(pwd)
fi
source $SCRIPTS_DIR/utils.sh


UPLOADS_REMOTE=git@github.com:faster-cpython/ideas.git
UPLOADS_BLOB_URL=https://github.com/faster-cpython/ideas/blob/main


function uploads-resolve-repo() {
    local datadir=$(resolve-data-dir $1)
    echo "$datadir/faster-cpython-ideas"
}

function uploads-ensure-repo() {
    local repodir=$1
    if [ -z "$repodir" -o "$repodir" == '-' ]; then
        repodir=$(uploads-resolve-repo $repodir)
    fi
    ensure-repo $UPLOADS_REMOTE $repodir

    echo "# make sure faster-cpython-ideas is clean and on the latest main"
    pushd-quiet $repodir
    (
    set -x
    git checkout main
    git pull
    )
    popd-quiet
}

function uploads-ensure-deps() {
    local repodir=$(uploads-resolve-repo $1)
    uploads-ensure-repo $repodir
}

function upload-file() {
    local datadir=$1
    local localfile=$2
    local remotefile=$3
    local msg=$4

    if [ -z "$remotefile" ]; then
        remotefile=$(basename "$localfile")
    fi

    if [ -z "$msg" ]; then
        msg='Add a data file.'
    fi

    local repodir=$(uploads-resolve-repo $datadir)
    uploads-ensure-repo $repodir
    pushd-quiet $repodir
    set -e
    (
    set -x
    cp "$localfile" "$remotefile"
    git add "$remotefile"
    git commit -m "$msg"
    git push
    )
    set +e
    popd-quiet
}

function uploads-resolve-url() {
    local remotefile=$1
    echo "$UPLOADS_BLOB_URL/$remotefile"
}


#######################################
# the script

if [ "$0" == "$BASH_SOURCE" ]; then
    datadir=$1
    localfile=$2
    remotefile=$3
    msg=$4

    if [ -z "$remotefile" ]; then
        remotefile=$(basename "$localfile")
    fi

    echo "### uploading $localfile ###"
    echo
    if ! upload-file "$datadir" "$localfile" "$remotefile" "$msg"; then
        fail "upload failed!"
    fi
    echo
    echo "# uploaded to $(uploads-resolve-url $remotefile)"
fi
