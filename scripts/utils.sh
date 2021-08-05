#!/bin/env bash

SCRIPTS_DIR=$(dirname $BASH_SOURCE)
if [ -z "$SCRIPTS_DIR" ]; then
    SCRIPTS_DIR=$(pwd)
fi


EC_TRUE=0
EC_FALSE=1

DEVNULL='/dev/null'


function print-err() {
    >&2 echo "ERROR: $1"
}

function fail() {
    print-err "$1"
    exit $EC_FALSE
}

function get-timestamp() {
    date --utc +"%Y%m%d_%H%M%S"
}

function pushd-quiet() {
    &>$DEVNULL pushd $1
}

function popd-quiet() {
    &>$DEVNULL popd
}

function ensure-dir() {
    local targetdir=$1
    if [ -z "$targetdir" ]; then
        fail "missing targetdir"
    fi
    (
    set -x
    mkdir -p "$targetdir"
    )
}


#############################
# data dir

PERF_DIR=$HOME/perf-data

function resolve-data-dir() {
    local datadir=$1
    if [ -z "$datadir" -o "$datadir" == '-' ]; then
        datadir=$PERF_DIR
    fi
    echo $datadir
}


#############################
# git

function ensure-repo() {
    local remote=$1
    local localdir=$2

    if [ -e $localdir ]; then
        # XXX Check it?
        :
    else
        ensure-dir $(dirname "$localdir")
        (
        set -x
        git clone "$remote" "$localdir"
        )
    fi
}

function is-repo-clean() {
    pushd-quiet $1
    git diff-index --quiet HEAD
    local rc=$?
    popd-quiet
    return $rc
}

function get-repo-info() {
    pushd-quiet $1
    (
    local branch=$(git rev-parse --abbrev-ref HEAD)
    if [ "$branch" == 'HEAD' ]; then
        branch=""
    fi
    local revision=$(git rev-parse --short HEAD)
    local clean='false'
    if git diff-index --quiet HEAD; then
        clean='true'
    fi
    )
    popd-quiet

    cat << EOF
{
    "branch": "$branch",
    "revision": "$revision",
    "clean": $clean
}
EOF
}

function parse-git-remote-url() {
    local raw=$1

    local rc=$EC_TRUE
    local url=
    local protocol=
    local service=
    local path=
    if [[ $raw == "git@"* ]]; then
        # e.g. git@github.com:python/cpython.git
        if [[ $raw != *".git" ]]; then
            # Add the forgotten suffix.
            # XXX Is the suffix standard enough to do this?
            raw="$raw.git"
        fi
        url=$raw
        raw=${raw:4}
        raw=${raw%.git}

        protocol='ssh'
        service=${raw%:*}
        path=${raw#*:}
    elif [[ $raw == "https://" ]]; then
        # e.g. https://github.com/python/cpython
        url=$raw
        raw=${raw:8}
        raw=${raw%/}

        protocol='https'
        service=${raw%/*/*}
        path=${raw#*/}
    else
        url=$raw
        rc=$EC_FALSE
    fi
    local org=${path%/*}
    local repo=${path#*/}

cat << EOF
{
    "url": "$url",
    "protocol": "$protcol",
    "service": "$service",
    "org": "$org",
    "repo": "$repo"
}
EOF
    return $rc
}

function parse-git-remote() {
    local raw=$1
    local infer='yes'

    local rc=$EC_TRUE
    local name=
    local url=
    case $raw in
      :https://*|:git@*)
        url=${raw:1}
        ;;
      https://*|git@*)
        url=$raw
        ;;
      *:*)
        name=$(echo $raw | cut -d':' -f 1)
        url=${raw#*:}
        ;;
      *)
        name=$raw
        if [ -z "$raw" ]; then
            rc=$EC_FALSE
        fi
        ;;
    esac

    if [ -z "$name" ]; then
        if [ $infer == 'yes' -a -n "$url" ]; then
            # Use the URL's org as the remote name.
            local url_info=$(parse-git-remote-url $url)
            name=$(echo $url_info | jq -r '.org')
        fi
    fi

cat << EOF
{
    "name": "$name",
    "url": "$url"
}
EOF
    return $rc
}

function ensure-git-remote() {
    local repodir=$1
    local name=$2
    local url=$3

    if [ -z "$repodir" ]; then
        fail "missing repo dir"
    fi
    if [ -z "$name" ]; then
        if [ -z "$url" ]; then
            fail "missing remote name"
        fi
        name=$(parse-git-remote-url $url | jq '.org')
    fi

    if [ -z "$url" ]; then
        local remote_info=$(parse-git-remote $name)
        name=$(echo $remote_info | jq -r '.name')
        url=$(echo $remote_info | jq -r '.url')
        if [ -z "$url" ]; then
            # There is no URL, so verify it was already added.
            # XXX Try "https://github.com/$name/$repo" first?
            pushd-quiet $repodir
            2>$DEVNULL git remote get-url $name
            local res=$?
            popd-quiet
            if [ $res -ne $EC_TRUE ]; then
                print-err "no remote named '$name'"
                return $EC_FALSE
            fi
            echo $name
            return $EC_TRUE
        fi
    fi

    local res=
    pushd-quiet $repodir
    local url_actual=$(2>$DEVNULL git remote get-url $name)
    res=$?
    popd-quiet
    if [ $? -ne $EC_TRUE ]; then
        # It hasn't been added yet, so add it.
        pushd-quiet $repodir
        git remote add $name $url
        res=$?
        popd-quiet
    elif [ -z "$url_actual" ]; then
        fail 'this should never happen ($url_actual should always be set)'
    elif [ $url != $url_actual ]; then
        print-err "URL mismatch for remote '$name' ($url != $url_actual)"
        res=$EC_FALSE
    fi

    if [ $res -eq $EC_TRUE ]; then
        echo $name
    fi
    return $res
}

function is-git-ref() {
    local repo=$1
    local raw=$2

    pushd-quiet $repo
    local revision=$(git rev-parse $raw)
    popd-quiet
    if [[ $revision == "$raw"* ]]; then
        return $EC_FALSE
    fi
    return $EC_TRUE
}
