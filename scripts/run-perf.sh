#!/bin/env bash

# See: https://www.brendangregg.com/flamegraphs.html

TIMESTAMP=$(date --utc +"%Y%m%d-%H%M")

PERF_DIR=$HOME/perf-data

PYTHON_URL='https://github.com/python/cpython'
#PYTHON_REPO=$LOCAL_BENCH_DIR/repositories/cpython
PYTHON_REPO=$PERF_DIR/cpython
PYTHON_BRANCH='main'
PYTHON=$PYTHON_REPO/python

CONFIG_CACHE=$PERF_DIR/python-config.cache
RELEASE_OPTS=" \
    --enable-ipv6 \
"
DEBUG_OPTS=" \
    --with-pydebug \
    --with-system-expat \
    --with-system-ffi \
"

PYTHON_PLAIN="$PYTHON -c 'pass'"
PYTHON_NO_SITE="$PYTHON -S -c 'pass'"

MAX_STACK=1000
FREQUENCY=10000

FLAMEGRAPH_URL=https://github.com/brendangregg/FlameGraph
FLAMEGRAPH_REPO=$PERF_DIR/FlameGraph

FASTER_CPYTHON_URL=git@github.com:faster-cpython/ideas.git
FASTER_CPYTHON_REPO=$PERF_DIR/faster-cpython-ideas


function fail() {
    >&2 echo "ERROR: $1"
    exit 1
}

function ensure-cpython-deps() {
    if [ ! -e $PYTHON_REPO ]; then
        (
        set -x
        &>$output git clone $PYTHON_URL $PYTHON_REPO
        )
    fi
}

function build-python() {
    local verbose=$1

    # Make sure we have the build we want.
    &>/dev/null pushd $PYTHON_REPO
    if [ $verbose == "yes" ]; then
        (
        set -x
        git checkout main
        make distclean
        #./configure $config_opts
        ./configure \
            CFLAGS=-O0 \
            --prefix=/opt/python-main \
            $config_opts
            #--cache-file=$CONFIG_CACHE \
        make -j8 CFLAGS=-fno-omit-frame-pointer
        )
    else
        (
        set -x
        &>/dev/null git checkout main
        &>/dev/null make distclean
        &>/dev/null ./configure $config_opts
        &>/dev/null make -j8
        )
    fi
    &>/dev/null popd
}

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

function ensure-flamegraph-deps() {
    if [ ! -e $FLAMEGRAPH_REPO ]; then
        echo "# get flamegraph tools"
        (
        set -x
        git clone $FLAMEGRAPH_URL $FLAMEGRAPH_REPO
        )
    fi
}

function ensure-upload-deps() {
    if [ ! -e $FASTER_CPYTHON_REPO ]; then
        (
        set -x
        git clone $FASTER_CPYTHON_URL $FASTER_CPYTHON_REPO
        )
    fi
    echo "# make sure faster-cpython-ideas is clean and on the latest main"
    &>/dev/null pushd $FASTER_CPYTHON_REPO
    (
    set -x
    git checkout main
    git pull
    )
    &>/dev/null popd
}

function warm-up() {
    local cmd=$1

    for i in {1..4}; do
        (
        set -x
        $cmd
        )
    done
}

function run-perf() {
    local record=$1
    local frequency=$2
    local outfile=$3
    local cmd=$4
    local opts=

    case $record in
      plain-graph)
        extra_opts=" \
            --call-graph fp \
        "
            #-call-graph fp
            #-call-graph dwarf
            #--call-graph lbr
            #--all-kernel
            #--all-user
        ;;
      *)
        fail "unsupported $record"
    esac

    local sample_rate_orig=$(cat /proc/sys/kernel/perf_event_max_sample_rate)
    local max_stack_orig=$(cat /proc/sys/kernel/perf_event_max_stack)
    sysctl kernel.perf_event_max_sample_rate
    sysctl kernel.perf_event_max_stack
    (
    set -x
    sudo sysctl -w kernel.perf_event_max_sample_rate=$frequency
    sudo sysctl -w kernel.perf_event_max_stack=$MAX_STACK
    sudo perf record \
        --all-cpus \
        -F max \
        --output $outfile \
        $extra_opts \
        -- $cmd
        #--exclude-perf
    )
    (
    set -x
    sudo sysctl -w kernel.perf_event_max_sample_rate=$sample_rate_orig
    sudo sysctl -w kernel.perf_event_max_stack=$max_stack_orig
    )
}

function run-strace() {
    local mode=$1
    local outfile=$2
    local cmd=$3

    local opts=
    if [ -n "$outfile" ]; then
        opts="$opts --output $outfile"
    fi
    #strace --trace process ...  # proc mgmt
    #strace --trace file ...  # filename as arg
    #strace --trace network ...
    #strace --trace signal ...
    #strace --trace memory ...  # mem mapping
    #strace --trace stat ...
    case "$mode" in
      summary)
        opts="$opts -c"
        ;;
      full)
        # full, with elapsed time
        opts="$opts -T"
        ;;
      *)
        fail "unsupported $mode"
    esac

    strace $opts $cmd
}

function create-flamegraph() {
    local frequency=$1
    local datafile=$2
    local scriptfile=$3
    local foldedfile=$4
    local outfile=$5

    local sample_rate_orig=$(cat /proc/sys/kernel/perf_event_max_sample_rate)
    local max_stack_orig=$(cat /proc/sys/kernel/perf_event_max_stack)
    (
    set -x
    sudo sysctl -w kernel.perf_event_max_sample_rate=$frequency
    sudo sysctl -w kernel.perf_event_max_stack=$MAX_STACK
    sudo perf script \
        --max-stack $MAX_STACK \
        --input $datafile \
        > $scriptfile
    sudo sysctl -w kernel.perf_event_max_sample_rate=$sample_rate_orig
    sudo sysctl -w kernel.perf_event_max_stack=$max_stack_orig

    $FLAMEGRAPH_REPO/stackcollapse-perf.pl $scriptfile > $foldedfile
    $FLAMEGRAPH_REPO/flamegraph.pl $foldedfile > $outfile
    )
}

function upload-file() {
    local localfile=$1
    local remotefile=$2
    local msg=$3

    if [ -z "$msg" ]; then
        msg='Add a profiler data file.'
    fi

    &>/dev/null pushd $FASTER_CPYTHON_REPO
    (
    set -x
    cp $localfile $remotefile
    git add $remotefile
    git commit -m "$msg"
    git push
    )
    &>/dev/null popd
}

function show-perf-report() {
    local mode=$1
    local datafile=$2

    local extra_opts=
    #--column-widths 100
    case "$mode" in
      header)
        extra_opts=" \
            $extra_opts
            --header \
        "
        ;;
      all-symbols)
        extra_opts=" \
            $extra_opts
            --sort symbol \
            --fields overhead_children,overhead,sample,overhead_sys,overhead_us,comm,symbol \
            --sort sample \
        "
        ;;
      python-symbols)
        extra_opts=" \
            $extra_opts
            --sort symbol \
            --fields overhead_children,overhead,sample,overhead_sys,overhead_us,comm,symbol \
            --sort sample \
            --comm python \
        "
        ;;
      tree)
        extra_opts=" \
            --sort sample
        "
        ;;
      tree-comm)
        extra_opts=" \
            --fields overhead_children,overhead,sample,comm
            --sort sample
        "
        ;;
      shared-objects)
        extra_opts=" \
            --fields overhead_children,overhead,sample,dso \
            --sort sample \
        "
        ;;
      *)
        fail "unsupported $mode"
        ;;
    esac

    (
    set -x
    sudo perf report \
        --input $datafile \
        --max-stack 10000 \
        $extra_opts
    )
}


#############################
# Parse the CLI args.

# general opts
verbose="yes"
op=
opset="no"
tool=
# Python opts
needpython="no"
rebuild=
debug="no"
site="yes"
# perf opts
frequency=$FREQUENCY
reuse="no"
record=
report=
flamegraph="no"
upload=
# strace opts
strace=

PERF_OPS=" \
    flamegraph \
    report \
    symbols \
    python-symbols \
    tree \
    tree-comm \
    shared-objects \
"

function check-opt() {
    local expected=$1
    local opt=$2

    if [ -n "$expected" -a "$expected" != "*" ]; then
        if [ "$op" == "$expected" ]; then
            return 0
        elif [ "$expected" == "flamegraph" ]; then
            if [ -z "$op" -o "$op" == "$PERF_OPS" ]; then
                set-op "flamegraph"
                return 0
            fi
        elif [ "$expected" == "perf" ]; then
            if [ -z "$op" -o "$op" == "$PERF_OPS" ]; then
                op="$PERF_OPS"
                return 0
            elif [[ " $PERF_OPS " == *" $expected "* ]]; then
                return 0
            fi
        fi
    fi
    fail "invalid option $opt"
}

function set-perf-op() {
    local arg=$1

    record="plain-graph"

    case "$arg" in
      flamegraph)
        flamegraph="yes"
        if [ -z "$upload" ]; then
            upload="yes"
        fi
        report=
        ;;
      report)
        report="header"
        ;;
      symbols)
        report="all-symbols"
        ;;
      python-symbols)
        report="python-symbols"
        ;;
      tree)
        report="tree"
        ;;
      tree-comm)
        report="tree-comm"
        ;;
      shared-objects)
        report="shared-objects"
        ;;
      *)
        fail "unsupported op $arg"
        ;;
    esac
}

function set-op() {
    local arg=$1

    if [ "$op" == "$PERF_OPS" ]; then
        if [[ " $PERF_OPS " == *" $arg "* ]]; then
            op=
        fi
    elif [ $opset == "no" -a "$arg" == "$op" ]; then
        op=
    fi
    if [ -n "$op" ]; then
        fail "unexpected arg $arg"
    fi

    op="$arg"
    case "$arg" in
      strace)
        strace="full"
        tool="strace"
        needpython="yes"
        ;;
      *)
        set-perf-op $arg
        tool="perf"
        needpython="yes"
        ;;
    esac

    if [ -z "$upload" ]; then
        upload="no"
    fi
}

while test $# -gt 0 ; do
  arg=$1
  shift
  case $arg in
    --help|-h)
      cat << EOF
$0 [COMMON_OPTIONS] [OP_OPTIONS] [OP]

Common Options:
  -h, --help
  -q, --quiet
  -v, --verbose
  --rebuild, --no-rebuild
  --debug | --release
  --site, --no-site

Operations (perf):
  <common>
    --frequency HERTZ
    --reuse
  flamegraph (default)
    --upload, --no-upload
    --report
  report
  symbols
  python-symbols
  tree
  tree-comm
  shared-objects

Operations (other):
  strace
    --summary
EOF
      exit 0
      ;;
    --verbose)
      verbose="yes"
      ;;
    --quiet)
      verbose="no"
      ;;

    # Python options
    --rebuild)
      rebuild="yes"
      ;;
    --no-rebuild)
      rebuild="no"
      ;;
    --debug)
      debug="yes"
      ;;
    --release)
      debug="no"
      ;;
    --site)
      site="yes"
      ;;
    --no-site)
      site="no"
      ;;

    # perf options
    --freq|--frequency)
      check-opt perf $arg
      frequency="$1"
      shift
      ;;
    --reuse)
      check-opt perf $arg
      reuse="yes"
      ;;
    --report)
      check-opt flamegraph $arg
      report="header"
      ;;
    --upload)
      check-opt flamegraph $arg
      upload="yes"
      ;;
    --no-upload)
      check-opt flamegraph $arg
      upload="no"
      ;;

    # strace options
    --summary)
      check-opt strace $arg
      strace="summary"
      ;;

    -*)
      check-opt "*" $arg
      ;;
    *)
      set-op $arg
      opset="yes"
      ;;
  esac
done

# set defaults
if [ -z "$op" -o "$op" == "$PERF_OPS" ]; then
    set-op "flamegraph"
    opset="yes"
fi

# extrapolate perf-related variables
tags=""
if [ $site == "no" ]; then
    tags="$tags-nosite"
fi
if [ $tool == "perf" -o -n "$record" ]; then
    useperf="yes"
    perfid="$op-$frequency"
    perf_data_file="$PERF_DIR/perf-$perfid$tags.data"
    perf_script_file="$PERF_DIR/out-$perfid$tags.perf"
    if [ $flamegraph == "yes" ]; then
        perf_folded_file="$PERF_DIR/out-$perfid$tags.folded"
        flamegraph_file="$PERF_DIR/$perfid$tags.svg"
        if [ $upload == "yes" ]; then
            #upstream_file="flamegraph-$TIMESTAMP$tags.svg"
            upstream_file="flamegraphs/freq$frequency$tags.svg"
        fi
    fi

    if [ $reuse == "yes" ]; then
        if [ -e $perf_data_file ]; then
            record=
        fi
    fi
else
    useperf="no"
fi

# extrapolate strace-related variables
if [ $tool == "strace" ]; then
    strace_data_file=
    #if [ "$strace" == "full" ]; then
    #    strace_data_file="$PERF_DIR/strace$tags.data"
    #fi
fi

# extrapolate Python-related variables
if [ $debug == "yes" ]; then
    config_opts="$DEBUG_OPTS"
else
    config_opts="$RELEASE_OPTS"
fi
if [ $site == "yes" ]; then
    python_command=$PYTHON_PLAIN
else
    python_command=$PYTHON_NO_SITE
fi
if [ $tool == "perf" -a -z "$record" ]; then
    needpython="no"
fi
if [ -z "$rebuild" ]; then
    rebuild="$needpython"
elif [ $needpython == "yes" ]; then
    if [ ! -e $PYTHON ]; then
        rebuild="yes"
    fi
fi
warmup=$needpython


#############################
# the script

echo
echo "*** profiling $python_command ***"
echo
echo "# data files in $PERF_DIR"
echo
mkdir -p $PERF_DIR

# ensure dependencies
if [ $rebuild == "yes" ]; then
    ensure-cpython-deps
    echo
    echo "==== building Python ===="
    echo
    build-python $verbose
fi
if [ $useperf == "yes" ]; then
    ensure-perf-deps
    if [ $flamegraph == "yes" ]; then
        ensure-flamegraph-deps
    fi
fi
if [ $upload == "yes" ]; then
    ensure-upload-deps
fi

# run the profiler
if [ $warmup == "yes" ]; then
    echo
    echo "==== warming up the disk cache ===="
    echo
    warm-up "$python_command"
fi
if [ -n "$record" ]; then
    echo
    echo "==== generating profile data with perf ===="
    echo
    run-perf $record $frequency $perf_data_file "$python_command"
fi
if [ -n "$strace" ]; then
    echo
    echo "==== generating profile data with strace ===="
    echo
    run-strace $strace "$strace_data_file" "$python_command"
fi

# generate any other data files
if [ $flamegraph == "yes" ]; then
    echo
    echo "==== generating the flame graph ===="
    echo
    create-flamegraph \
        $frequency \
        $perf_data_file \
        $perf_script_file \
        $perf_folded_file \
        $flamegraph_file
    if [ $upload == "yes" ]; then
        echo
        echo "==== upload the SVG file ===="
        echo
        upload-file $flamegraph_file $upstream_file 'Add a flame graph SVG file.'
    fi
    echo
    echo "*** flame graph located at $flamegraph_file ***"
fi

# do any extra reporting
if [ -n "$report" ]; then
    # Use perf to review the results.
    show-perf-report $report $perf_data_file
fi
