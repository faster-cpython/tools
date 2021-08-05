#!/bin/env bash

SCRIPTS_DIR=$(dirname $BASH_SOURCE)
if [ -z "$SCRIPTS_DIR" ]; then
    SCRIPTS_DIR=$(pwd)
fi
source $SCRIPTS_DIR/utils.sh


CPYTHON_URL='https://github.com/python/cpython'
CPYTHON_REPO=$PERF_DIR/cpython
PYTHON=$CPYTHON_REPO/python

PYTHON_PREFIX=/opt/python-perf-tools-cpython-build
CONFIG_CACHE=$PERF_DIR/python-config.cache
BUILD_INFO_FILE=$PERF_DIR/python-build-info


function cpython-ensure-deps() {
    sudo apt install -y \
      build-essential \
      zlib1g-dev \
      libbz2-dev \
      liblzma-dev \
      libncurses5-dev \
      libreadline6-dev \
      libsqlite3-dev \
      libssl-dev \
      libgdbm-dev \
      liblzma-dev \
      tk-dev \
      lzma \
      lzma-dev \
      libgdbm-dev \
      libgdbm-compat-dev \
      libffi-dev \

}


# repo

function cpython-ensure-repo() {
    if [ ! -e $CPYTHON_REPO ]; then
        (
        set -x
        &>$DEVNULL git clone $CPYTHON_URL $CPYTHON_REPO
        )
    fi
}

#function cpython-ensure-revision() {
#    local revision=$1
#    local remote=$2
#
#    if ! is-repo-clean $CPYTHON_REPO; then
#        fail "CPython repo isn't clean"
#    fi
#
#    if [ -n "$remote" ]; then
#        if is-git-ref $revision; then
#            local remote_orig=$remote
#            remote=$(ensure-git-remote $CPYTHON_REPO $remote)
#            if [ $? -ne $EC_TRUE ]; then
#                fail "bad remote '$remote_orig'"
#            else
#                git checkout $remote/$revision
#            fi
#        fi
#    else
#        git checkout $revision
#    fi
#}


# build info

function cpython-get-config-opts() {
    local build=$1
    local optlevel=$2
    local prefix=$3
    if [ -z "$prefix" -o "$prefix" == 'yes' ]; then
        prefix=$PYTHON_PREFIX
    fi

    echo -n " --cache-file=$CONFIG_CACHE"
    echo -n " --prefix=$prefix"
    echo -n ' --enable-ipv6'

    case "$build" in
      release)
        ;;
      debug)
        echo -n ' --with-pydebug'
        echo -n ' --with-system-expat'
        echo -n ' --with-system-ffi'
        ;;
      *)
        fail "unsupported build '$build'"
        ;;
    esac

    if [ -n "$optlevel" ]; then
        echo -n " CFLAGS=-O$optlevel"
    fi

    # Add the trailing LF.
    echo ""
}

function cpython-read-config-opts() {
    if [ ! -e $BUILD_INFO_FILE ]; then
        return $EC_FALSE
    fi
    jq -r '.config_opts' $BUILD_INFO_FILE
    return $EC_TRUE
}

function cpython-get-build-id() {
    local build=$1
    local optlevel=$2
    local revision=$3

    if [ -z "$revision" ]; then
        pushd-quiet $CPYTHON_REPO
        local revision=$(git rev-parse --short HEAD)
        popd-quiet
    fi

    local buildid="$revision-$build"
    if [ -n "optlevel" ]; then
        buildid="$buildid-opt$optlevel"
    fi
    echo $buildid
}

function cpython-read-build-id() {
    if [ ! -e $BUILD_INFO_FILE ]; then
        return $EC_FALSE
    fi
    jq '.buildid' $BUILD_INFO_FILE
    return $EC_TRUE
}

function cpython-ensure-build-info() {
    local buildid=$1
    local build=$2
    local optlevel=$3
    local prefix=$4
    local revision=$5
    local configopts=$6

    local branch=
    if [ -z "$revision" ]; then
        local repoinfo=$(get-repo-info $CPYTHON_REPO)
        revision=$(echo $repoinfo | jq '.revision')
        branch=$(echo $repoinfo | jq '.branch')
    else
        pushd-quiet $CPYTHON_REPO
        branch=$(git rev-parse --abbrev-ref HEAD)
        popd-quiet
        if [ "$branch" == 'HEAD' ]; then
            branch=""
        fi
    fi

    if [ -z "$buildid" ]; then
        buildid=$(cpython-get-build-id "$build" "$optlevel" "$revision")
    fi
    local lastid=$(cpython-read-build-id)
    if [ $? -eq $EC_TRUE -a $buildid == $lastid ]; then
        return $EC_TRUE
    fi

    if [ -z "$configopts" ]; then
        configopts=$(cpython-get-config-opts "$build" "$optlevel" "$prefix")
    fi

    cat << EOF > $BUILD_INFO_FILE
{
    "buildid": "$buildid",
    "build": "$build",
    "config_opts": "$configopts",
    "branch": "$branch",
    "revision": "$revision"
}
EOF
    return $EC_FALSE
}


# building

function cpython-config-opts-changed() {
    local build=$1
    local optlevel=$2
    local prefix=$3

    # XXX

    return $EC_FALSE
}

function cpython-build() {
    local build=$1
    local optlevel=$2
    local prefix=$3
    local verbose=$4
    local force=$5

    pushd-quiet $CPYTHON_REPO
    local revision=$(git rev-parse --short HEAD)
    popd-quiet

    local buildid=$(cpython-get-build-id "$build" "$optlevel" "$revision")

    if [ "$force" != 'yes' ]; then
        if [ -e $PYTHON ]; then
            local buildid_old=$(cpython-read-build-id)
            if [ "$buildid" == "$buildid_old" ]; then
                # The build is up-to-date.
                return $EC_FALSE
            fi
        fi
    fi

    if ! is-repo-clean $CPYTHON_REPO; then
        fail "CPython repo isn't clean"
    fi

    if cpython-config-opts-changed "$build" "$optlevel" "$prefix"; then
        pushd-quiet $CPYTHON_REPO
        if [ "$verbose" == "yes" ]; then
            (
            set -x
            make distclean
            2>$DEVNULL rm $CONFIG_CACHE
            )
        else
            (
            set -x
            &>$DEVNULL make distclean
            &>$DEVNULL rm $CONFIG_CACHE
            )
        fi
        popd-quiet
    fi

    local configopts=$(cpython-get-config-opts "$build" "$optlevel" "$prefix")
    cpython-ensure-build-info "$buildid" "$build" "$optlevel" "$prefix" "$revision" "$configopts"

    # Run the build.
    echo
    echo "==== building Python ===="
    echo
    pushd-quiet $CPYTHON_REPO
    if [ "$verbose" == "yes" ]; then
        (
        set -x
        ./configure $configopts
        make -j8 'CFLAGS=-fno-omit-frame-pointer -pg'
        )
    else
        (
        set -x
        &>$DEVNULL ./configure $configopts
        &>$DEVNULL make -j8 'CFLAGS=-fno-omit-frame-pointer -pg'
        )
    fi
    popd-quiet
    return $EC_TRUE
}


#######################################
# the script

if [ "$0" == "$BASH_SOURCE" ]; then
    #cpython-ensure-deps
    #cpython-ensure-repo
    cpython-build "$@"
fi
