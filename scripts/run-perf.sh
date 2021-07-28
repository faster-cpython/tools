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


#############################
# Parse the CLI args.

op=
flamegraph="no"
report="no"
upload=
frequency=$FREQUENCY
site="yes"
record="yes"
rebuild="yes"
debug="no"
verbose="yes"
while test $# -gt 0 ; do
  arg=$1
  shift
  case $arg in
    --help|-h)
      cat << EOF
$0 [COMMON_OPTIONS] [OP_OPTIONS] [OP]

Operations:
  flamegraph (default)
    --upload, --no-upload
    --report
  report
  symbols
  python-symbols
  tree
  tree-comm
  shared-objects

Common Options:
  -h, --help
  -q, --quiet
  -v, --verbose
  --rebuild, --no-rebuild
  --debug | --release
  --site, --no-site
  --frequency HERTZ
  --reuse
EOF
      exit 0
      ;;
    --reuse)
      record="no"
      ;;
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
    --freq|--frequency)
        frequency="$1"
        shift
        ;;
    --report)
      report="yes"
      ;;
    --upload)
      upload="yes"
      ;;
    --no-upload)
      upload="no"
      ;;
    --verbose)
      verbose="yes"
      ;;
    --quiet)
      verbose="no"
      ;;
    -*)
      >&2 echo "ERROR: invalid option $arg"
      exit 1
      ;;
    *)
      if [ -z "$op" ]; then
        op="$arg"
        case $arg in
          flamegraph)
            flamegraph="yes"
            ;;
          report|symbols|python-symbols|tree|tree-comm|shared-objects)
            report="yes"
            ;;
          *)
            >&2 echo "ERROR: unsupported op $arg"
            exit 1
            ;;
        esac
      else
        >&2 echo "ERROR: unexpected arg $arg"
        exit 1
      fi
      ;;
  esac
done
if [ -z "$op" ]; then
    op="flamegraph"
    flamegraph="yes"
fi

if [ -z "$upload" ]; then
    upload="$flamegraph"
elif [ "$upload" == "yes" -a $flamegraph == "no" ]; then
    >&2 echo "ERROR: --upload not supported for $op"
    exit 1
fi

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

case $op in
  flamegraph)
    record_opts=" \
        --call-graph fp \
    "
        #-call-graph fp
        #-call-graph dwarf
        #--call-graph lbr
        #--all-kernel
        #--all-user
    report_opts=" \
        --header \
    "
    ;;
  report)
    record_opts=" \
        --call-graph fp \
    "
    report="yes"
    report_opts=" \
        --header \
    "
    ;;
  symbols)
    record_opts=" \
        --call-graph fp \
    "
    report="yes"
    report_opts=" \
        --sort symbol \
        --fields overhead_children,overhead,sample,overhead_sys,overhead_us,comm,symbol \
        --sort sample \
    "
    ;;
  python-symbols)
    record_opts=" \
        --call-graph fp \
    "
    report="yes"
    report_opts=" \
        --sort symbol \
        --fields overhead_children,overhead,sample,overhead_sys,overhead_us,comm,symbol \
        --sort sample \
        --comm python \
    "
    ;;
  tree)
    record_opts=" \
        --call-graph fp \
    "
    report="yes"
    report_opts=" \
        --sort sample
    "
    ;;
  tree-comm)
    record_opts=" \
        --call-graph fp \
    "
    report="yes"
    report_opts=" \
        --fields overhead_children,overhead,sample,comm
        --sort sample
    "
    ;;
  shared-objects)
    record_opts=" \
        --call-graph fp \
    "
    report="yes"
    report_opts=" \
        --fields overhead_children,overhead,sample,dso \
        --sort sample \
    "
    ;;
esac

perfid="$op-$frequency"
tags=""
if [ $site == "no" ]; then
    tags="$tags-nosite"
fi
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

if [ $record == "no" ]; then
    if [ ! -e $perf_data_file ]; then
        record="yes"
    fi
fi
if [ $record == "yes" ]; then
    rebuild="no"
elif [ $rebuild == "no" ]; then
    if [ ! -e $PYTHON ]; then
        rebuild="yes"
    fi
fi


#############################
# run the script

echo
echo "*** profiling $python_command ***"
echo
echo "# data files in $PERF_DIR"
echo
mkdir -p $PERF_DIR

# Ensure dependencies.
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
if [ $flamegraph == "yes" ]; then
    if [ ! -e $FLAMEGRAPH_REPO ]; then
        echo "# get flamegraph tools"
        (
        set -x
        git clone $FLAMEGRAPH_URL $FLAMEGRAPH_REPO
        )
    fi
fi
if [ $upload == "yes" ]; then
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
fi
if [ $rebuild == "yes" ]; then
    if [ ! -e $PYTHON_REPO ]; then
        (
        set -x
        &>$output git clone $PYTHON_URL $PYTHON_REPO
        )
    fi

    # Make sure we have the build we want.
    echo
    echo "==== building Python ===="
    echo
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
fi

if [ $record == "yes" ]; then
    # Warm up the disk cache.
    echo
    echo "==== warming up the disk cache ===="
    echo
    for i in {1..4}; do
        (
        set -x
        $python_command
        )
    done

    echo
    echo "==== generating the profile data ===="
    echo

    if [ $flamegraph == "no" -a $report != "yes" ]; then
        >2 echo "not finished"; exit 1  # XXX
    fi

    sample_rate_orig=$(cat /proc/sys/kernel/perf_event_max_sample_rate)
    max_stack_orig=$(cat /proc/sys/kernel/perf_event_max_stack)
    sysctl kernel.perf_event_max_sample_rate
    sysctl kernel.perf_event_max_stack
    (
    set -x
    sudo sysctl -w kernel.perf_event_max_sample_rate=$frequency
    sudo sysctl -w kernel.perf_event_max_stack=$MAX_STACK
    sudo perf record \
        $record_opts \
        --all-cpus \
        -F max \
        --output $perf_data_file \
        -- $python_command
        #--exclude-perf
    )
    (
    set -x
    sudo sysctl -w kernel.perf_event_max_sample_rate=$sample_rate_orig
    sudo sysctl -w kernel.perf_event_max_stack=$max_stack_orig
    )
fi

if [ $flamegraph == "yes" ]; then
    echo
    echo "==== generating the flame graph ===="
    echo

    sample_rate_orig=$(cat /proc/sys/kernel/perf_event_max_sample_rate)
    max_stack_orig=$(cat /proc/sys/kernel/perf_event_max_stack)
    (
    set -x
    sudo sysctl -w kernel.perf_event_max_sample_rate=$frequency
    sudo sysctl -w kernel.perf_event_max_stack=$MAX_STACK
    sudo perf script \
        --max-stack $MAX_STACK \
        --input $perf_data_file > $perf_script_file
    sudo sysctl -w kernel.perf_event_max_sample_rate=$sample_rate_orig
    sudo sysctl -w kernel.perf_event_max_stack=$max_stack_orig

    $FLAMEGRAPH_REPO/stackcollapse-perf.pl $perf_script_file > $perf_folded_file
    $FLAMEGRAPH_REPO/flamegraph.pl $perf_folded_file > $flamegraph_file
    )
    
    if [ $upload == "yes" ]; then
        echo
        echo "==== upload the SVG file ===="
        echo
        &>/dev/null pushd $FASTER_CPYTHON_REPO
        (
        set -x
        cp $flamegraph_file $upstream_file
        git add $upstream_file
        git commit -m 'Add a flame graph SVG file.'
        git push
        )
        &>/dev/null popd
    fi
    
    echo
    echo "*** flame graph located at $flamegraph_file ***"
fi

if [ $report == "yes" ]; then
    # Use perf to review the results.
    (
    set -x
    sudo perf report \
        --input $perf_data_file \
        --max-stack 10000 \
        $report_opts
    )
    #    --column-widths 100
fi
