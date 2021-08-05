#!/bin/env bash

SCRIPTS_DIR=$(dirname $BASH_SOURCE)
if [ -z "$SCRIPTS_DIR" ]; then
    SCRIPTS_DIR=$(pwd)
fi
source $SCRIPTS_DIR/utils.sh
source $SCRIPTS_DIR/cpython.sh
source $SCRIPTS_DIR/upload.sh
source $SCRIPTS_DIR/run-profiler.sh


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
    local repodir=$(resolve-flamegraph-repo $1)  # from run-profiler.sh
    ensure-flamegraph-repo $repodir  # from run-profiler.sh
}


#######################################
# the script

if [ "$0" == "$BASH_SOURCE" ]; then
    local datadir='-'
    while test $# -gt 0 ; do
        arg=$1
        shift
        case $arg in
          --datadir)
            datadir=$1
            shift
            if [ -z "$datadir" ]; then
                datadir='-'
            fi
            ;;
          *)
            fail "unsupported arg $arg"
            ;;
        esac
    done
    datadir=$(resolve-data-dir $datadir)
    ensure-dir $datadir

    cpython-ensure-deps $datadir  # from cpython.sh
    ensure-perf-deps
    ensure-uftrace-deps
    ensure-strace-deps
    ensure-flamegraph-deps $datadir
    uploads-ensure-deps $datadir  # from upload.sh
fi
