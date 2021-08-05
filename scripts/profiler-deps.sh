#!/bin/env bash

SCRIPTS_DIR=$(dirname $BASH_SOURCE)
if [ -z "$SCRIPTS_DIR" ]; then
    SCRIPTS_DIR=$(pwd)
fi
source $SCRIPTS_DIR/utils.sh
source $SCRIPTS_DIR/cpython.sh
source $SCRIPTS_DIR/upload.sh


FLAMEGRAPH_URL=https://github.com/brendangregg/FlameGraph
# This is duplicated in run-profiler.sh.
FLAMEGRAPH_REPO=$PERF_DIR/FlameGraph


function ensure-perf-deps() {
    if ! &>/dev/null which perf; then
        echo "# install system dependencies"
        LINUX_KERNEL=$(uname -r)
        (
        set -x
        sudo apt install \
            linux-tools-common \
            linux-tools-generic \
            linux-cloud-tools-generic \
            linux-tools-$LINUX_KERNEL \
            linux-cloud-tools-$LINUX_KERNEL
        )
    fi
}

function ensure-uftrace-deps() {
    (
    set -x
    &>$DEVNULL sudo apt install -y uftrace
    )
    if [ $? -ne $EC_TRUE ]; then
        sudo apt install -y uftrace
        exit 1
    fi
}

function ensure-strace-deps() {
    (
    set -x
    sudo apt install \
        strace
    )
}

function ensure-flamegraph-deps() {
    if [ ! -e $FLAMEGRAPH_REPO ]; then
        echo "# get flamegraph tools"
        (
        set -x
        git clone $FLAMEGRAPH_URL $FLAMEGRAPH_REPO
        )
    fi
}


#######################################
# the script

if [ "$0" == "$BASH_SOURCE" ]; then
    cpython-ensure-deps  # from cpython.sh
    cpython-ensure-repo  # from cpython.sh
    ensure-perf-deps
    ensure-uftrace-deps
    ensure-strace-deps
    ensure-flamegraph-deps
    ensure-upload-deps  # from upload.sh
fi
