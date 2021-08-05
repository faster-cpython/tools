#!/bin/env bash

SCRIPTS_DIR=$(dirname $BASH_SOURCE)
if [ -z "$SCRIPTS_DIR" ]; then
    SCRIPTS_DIR=$(pwd)
fi
source $SCRIPTS_DIR/utils.sh


CPYTHON_URL='https://github.com/python/cpython'

PYTHON_PREFIX=/opt/python-perf-tools-cpython-build


function cpython-ensure-deps() {
    local repodir=$(cpython-resolve-repo $1)

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

    cpython-ensure-repo $repodir
}


# repo

function cpython-resolve-repo() {
    local datadir=$(resolve-data-dir $1)
    echo "$datadir/cpython"
}

function cpython-ensure-repo() {
    local repodir=$1
    if [ -z "$repodir" -o "$repodir" == '-' ]; then
        repodir=$(cpython-resolve-repo $repodir)
    fi
    ensure-repo $CPYTHON_URL $repodir
}

#function cpython-ensure-revision() {
#    local datadir=$1
#    local revision=$2
#    local remote=$3
#
#    local repodir=$(cpython-resolve-repo $datadir)
#    if ! is-repo-clean $repodir; then
#        fail "CPython repo isn't clean"
#    fi
#
#    if [ -n "$remote" ]; then
#        if is-git-ref $revision; then
#            local remote_orig=$remote
#            remote=$(ensure-git-remote $repodir $remote)
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

function cpython-resolve-exe() {
    local repodir=$(cpython-resolve-repo $1)
    echo "$repodir/python"
}

function cpython-resolve-config-cache() {
    local datadir=$(resolve-data-dir $1)
    echo "$datadir/python-config.cache"
}

function cpython-resolve-build-info() {
    local datadir=$(resolve-data-dir $1)
    echo "$datadir/python-build-info"
}

function cpython-get-config-opts() {
    local datadir=$1
    local build=$2
    local optlevel=$3
    local prefix=$4
    if [ -z "$build" -o "$build" == '-' ]; then
        build='release'
    fi
    if [ -z "$prefix" -o "$prefix" == '-' ]; then
        # XXX Allow not using "--prefix"?
        prefix=$PYTHON_PREFIX
    fi
    local configcache=$(cpython-resolve-config-cache $datadir)

    echo -n " --cache-file=$configcache"
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

    if [ -n "$optlevel" -a "$optlevel" != '-' ]; then
        echo -n " CFLAGS=-O$optlevel"
    fi

    # Add the trailing LF.
    echo ""
}

function cpython-read-config-opts() {
    local datadir=$1
    local infofile=$(cpython-resolve-build-info $datadir)
    if [ ! -e $infofile ]; then
        return $EC_FALSE
    fi
    jq -r '.config_opts' $infofile
    return $EC_TRUE
}

function cpython-get-build-id() {
    local datadir=$1
    local build=$2
    local optlevel=$3
    local revision=$4

    if [ -z "$revision" -o "$revision" == '-' ]; then
        local repodir=$(cpython-resolve-repo $datadir)
        cpython-ensure-repo $repodir
        pushd-quiet $repodir
        local revision=$(git rev-parse --short HEAD)
        popd-quiet
    fi

    local buildid="$revision-$build"
    if [ -n "$optlevel" -a "$optlevel" != '-' ]; then
        buildid="$buildid-opt$optlevel"
    fi
    echo $buildid
}

function cpython-read-build-id() {
    local datadir=$1
    local infofile=$(cpython-resolve-build-info $datadir)
    if [ ! -e $infofile ]; then
        return $EC_FALSE
    fi
    jq '.buildid' $infofile
    return $EC_TRUE
}

function cpython-ensure-build-info() {
    local datadir=$(resolve-data-dir $1)
    local buildid=$2
    local build=$3
    local optlevel=$4
    local prefix=$5
    local revision=$6
    local configopts=$7

    ensure-dir $datadir

    local repodir=$(cpython-resolve-repo $datadir)
    cpython-ensure-repo $repodir
    local branch=
    if [ -z "$revision" -o "$revision" == '-' ]; then
        local repoinfo=$(get-repo-info $repodir)
        revision=$(echo $repoinfo | jq '.revision')
        branch=$(echo $repoinfo | jq '.branch')
    else
        pushd-quiet $repodir
        branch=$(git rev-parse --abbrev-ref HEAD)
        popd-quiet
        if [ "$branch" == 'HEAD' ]; then
            branch=""
        fi
    fi

    if [ -z "$buildid" -o "$buildid" == '-' ]; then
        buildid=$(cpython-get-build-id "$datadir" "$build" "$optlevel" "$revision")
    fi
    local lastid=$(cpython-read-build-id $datadir)
    if [ $? -eq $EC_TRUE -a $buildid == $lastid ]; then
        return $EC_TRUE
    fi

    if [ -z "$configopts" -o "$configopts" == '-' ]; then
        configopts=$(cpython-get-config-opts "$datadir" "$build" "$optlevel" "$prefix")
    fi

    local infofile=$(cpython-resolve-build-info $datadir)
    cat << EOF > $infofile
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
    local datadir=$(resolve-data-dir $1)
    local build=$2
    local optlevel=$3
    local prefix=$4

    ensure-dir $datadir

    # XXX

    return $EC_FALSE
}

function cpython-build() {
    local datadir=$1
    local build=$2
    local optlevel=$3
    local prefix=$4
    local force=$5
    local verbose=$6

    local repodir=$(cpython-resolve-repo $datadir)
    cpython-ensure-repo $repodir
    pushd-quiet $repodir
    local revision=$(git rev-parse --short HEAD)
    popd-quiet

    local buildid=$(cpython-get-build-id "$datadir" "$build" "$optlevel" "$revision")

    local python=$(cpython-resolve-exe $datadir)
    if [ "$force" != 'yes' ]; then
        if [ -e $python ]; then
            local buildid_old=$(cpython-read-build-id $datadir)
            if [ "$buildid" == "$buildid_old" ]; then
                # The build is up-to-date.
                return $EC_FALSE
            fi
        fi
    fi

    echo "'$repodir'"
    if ! is-repo-clean $repodir; then
        fail "CPython repo isn't clean"
    fi

    if cpython-config-opts-changed "$datadir" "$build" "$optlevel" "$prefix"; then
        local configcache=$(cpython-resolve-config-cache $datadir)
        pushd-quiet $repodir
        if [ "$verbose" == "yes" ]; then
            (
            set -x
            make distclean
            2>$DEVNULL rm $configcache
            )
        else
            (
            set -x
            &>$DEVNULL make distclean
            &>$DEVNULL rm $configcache
            )
        fi
        popd-quiet
    fi

    local configopts=$(cpython-get-config-opts "$datadir" "$build" "$optlevel" "$prefix")
    cpython-ensure-build-info "$datadir" "$buildid" "$build" "$optlevel" "$prefix" "$revision" "$configopts"

    # Run the build.
    echo
    echo "==== building Python ===="
    echo
    pushd-quiet $repodir
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
    prep='no'
    datadir=$PERF_DIR
    build='release'
    optlevel='-'
    prefix=$PYTHON_PREFIX
    force='no'
    verbose='no'
    while test $# -gt 0 ; do
        arg=$1
        shift
        case $arg in
          --prep)
            prep='yes'
            ;;
          --datadir)
            datadir=$1
            shift
            if [ -z "$datadir" -o "$datadir" == '-' ]; then
                datadir=$PERF_DIR
            fi
            ;;
          --build)
            build=$1
            shift
            if [ -z "$build" -o "$build" == '-' ]; then
                build='release'
            fi
            ;;
          --optlevel)
            optlevel=$1
            shift
            if [ -z "$optlevel" ]; then
                optlevel='-'
            fi
            ;;
          --prefix)
            prefix=$1
            shift
            if [ -z "$prefix" -o "$prefix" == '-' ]; then
                prefix=$PYTHON_PREFIX
            fi
            ;;
          --force)
            force='yes'
            ;;
          --verbose)
            verbose='yes'
            ;;
          *)
            fail "unsupported arg $arg"
            ;;
        esac
    done

    if [ "$prep" == 'yes' ]; then
        cpython-ensure-deps $datadir
    fi
    cpython-build "$datadir" "$build" "$optlevel" "$prefix" "$force" "$verbose"
fi
