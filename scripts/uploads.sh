#!/bin/env bash

SCRIPTS_DIR=$(dirname $BASH_SOURCE)
if [ -z "$SCRIPTS_DIR" ]; then
    SCRIPTS_DIR=$(pwd)
fi
source $SCRIPTS_DIR/utils.sh


UPLOADS_REMOTE=git@github.com:faster-cpython/ideas.git
UPLOADS_BLOB_URL=https://github.com/faster-cpython/ideas/blob/main


function resolve-uploads-repo() {
    local datadir=$1
    if [ -z "$datadir" -o "$datadir" == '-' ]; then
        datadir=$PERF_DIR
    fi
    echo "$datadir/faster-cpython-ideas"
}

function ensure-uploads-deps() {
    local datadir=$1
    local repodir=$(resolve-uploads-repo $datadir)
    if [ ! -e $repodir ]; then
        (
        set -x
        git clone $UPLOADS_REMOTE $repodir
        )
    fi
    echo "# make sure faster-cpython-ideas is clean and on the latest main"
    pushd-quiet $repodir
    (
    set -x
    git checkout main
    git pull
    )
    popd-quiet
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

    local repodir=$(resolve-uploads-repo $datadir)
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

function get-upload-url() {
    local remotefile=$1
    echo "$UPLOADS_BLOB_URL/$remotefile"
}


#######################################
# the script

if [ "$0" == "$BASH_SOURCE" ]; then
    localfile=$1
    remotefile=$2
    msg=$3

    if [ -z "$remotefile" ]; then
        remotefile=$(basename "$localfile")
    fi

    echo "### uploading $localfile ###"
    echo
    if ! upload-file "$localfile" "$remotefile" "$msg"; then
        fail "upload failed!"
    fi
    echo
    echo "# uploaded to $(get-upload-url $remotefile)"
fi
