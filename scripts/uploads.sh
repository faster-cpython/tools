#!/bin/env bash

SCRIPTS_DIR=$(dirname $BASH_SOURCE)
if [ -z "$SCRIPTS_DIR" ]; then
    SCRIPTS_DIR=$(pwd)
fi
source $SCRIPTS_DIR/utils.sh


FASTER_CPYTHON_URL=git@github.com:faster-cpython/ideas.git
FASTER_CPYTHON_BLOB_URL=https://github.com/faster-cpython/ideas/blob/main
FASTER_CPYTHON_REPO=$PERF_DIR/faster-cpython-ideas


function ensure-upload-deps() {
    if [ ! -e $FASTER_CPYTHON_REPO ]; then
        (
        set -x
        git clone $FASTER_CPYTHON_URL $FASTER_CPYTHON_REPO
        )
    fi
    echo "# make sure faster-cpython-ideas is clean and on the latest main"
    pushd-quiet $FASTER_CPYTHON_REPO
    (
    set -x
    git checkout main
    git pull
    )
    popd-quiet
}

function upload-file() {
    local localfile=$1
    local remotefile=$2
    local msg=$3

    if [ -z "$remotefile" ]; then
        remotefile=$(basename "$localfile")
    fi

    if [ -z "$msg" ]; then
        msg='Add a data file.'
    fi

    pushd-quiet $FASTER_CPYTHON_REPO
    (
    set -x
    cp "$localfile" "$remotefile"
    git add "$remotefile"
    git commit -m "$msg"
    git push
    )
    popd-quiet
}

function get-upload-url() {
    local remotefile=$1
    echo "$FASTER_CPYTHON_BLOB_URL/$remotefile"
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
    upload-file "$localfile" "$remotefile" "$msg"
    echo
    echo "# uploaded to $(get-upload-url $remotefile)"
fi
