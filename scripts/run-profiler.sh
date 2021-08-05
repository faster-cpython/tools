#!/bin/env bash

_NOTES=\
'''
Be sure to run the "profiler-deps.sh" script before running this one.


## reference ##

* perf
   * ...
* flamegraphs
   * https://www.brendangregg.com/flamegraphs.html
* strace
   * ...


## useful profiling commands ##

PYTHON=$PERF_DIR/cpython/python


### perf ###

# Generating profile data is just a single command, since it requires
# changing various kernel settings.  Hence the "run-perf" function below.

# generate profile data for full Python startup
DATAFILE=$(run-perf "" "" "$PYTHON -c pass")

# generate profile data for minimal Python startup
DATAFILE=$(run-perf "" "-nosite" "$PYTHON -S -c pass")

# The following present previously generated perf data:

# just the header
perf report --input $DATAFILE --max-stack 10000 \
            --header

# all symbols
perf report --input $DATAFILE --max-stack 10000 \
            --sort symbol \
            --fields overhead_children,overhead,sample,overhead_sys,overhead_us,comm,symbol \
            --sort sample

# only Python symbols
perf report --input $DATAFILE --max-stack 10000 \
            --sort symbol \
            --fields overhead_children,overhead,sample,overhead_sys,overhead_us,comm,symbol \
            --sort sample \
            --comm python \

# call tree (ungrouped)
perf report --input $DATAFILE --max-stack 10000 \
            --sort sample

# call tree grouped by command
perf report --input $DATAFILE --max-stack 10000 \
            --fields overhead_children,overhead,sample,comm
            --sort sample

# call tree grouped by shared objects
perf report --input $DATAFILE --max-stack 10000 \
            --fields overhead_children,overhead,sample,dso \
            --sort sample \


### uftrace ###

# show where time was spent
uftrace report --data $DATADIR

# show the call graph
uftrace graph --data $DATADIR


### strace ###

# generate a summary after profiling full Python startup
strace -c $PYTHON -c pass

# generate profile data for full Python startup
strace -T $PYTHON -c pass

# various filters
strace --trace process ...  # proc mgmt
strace --trace file ...  # filename as arg
strace --trace network ...
strace --trace signal ...
strace --trace memory ...  # mem mapping
strace --trace stat ...
'''

SCRIPTS_DIR=$(dirname $BASH_SOURCE)
if [ -z "$SCRIPTS_DIR" ]; then
    SCRIPTS_DIR=$(pwd)
fi
source $SCRIPTS_DIR/utils.sh
source $SCRIPTS_DIR/uploads.sh


# This is duplicated in profiler-deps.sh.
FLAMEGRAPH_REPO=$PERF_DIR/FlameGraph

# CPython's C stacks can get pretty deep.
MAX_STACK=1000
PERF_FREQUENCY=99999  # ~100 khz


#######################################
# data files

DATAFILE_SUFFIX='.data'

NO_FREQUENCY='-'


function get-data-loc() {
    local tool=$1
    local tags=$2

    if [ -n "$tags" ]; then
        if [[ $tags != '-'* ]]; then
            tags=${tags:1}
        fi
    fi

    if [ "$tool" == 'perf' ]; then
        local frequency=$3
        if [ "$frequency" != $NO_FREQUENCY ]; then
            if [ -z "$frequency" ]; then
                frequency=$PERF_FREQUENCY
            fi
            tags="-freq$frequency$tags"
        fi
    elif [ "$tool" == 'uftrace' ]; then
        :
    else
        fail "unsupported tool '$tool'"
    fi
    echo "$PERF_DIR/$tool$tags$DATAFILE_SUFFIX"
}

function extract-datafile-fullid() {
    local datafile=$1

    local name=$(basename "$datafile")
    if [[ $name != *"$DATAFILE_SUFFIX" ]]; then
        print-err "does not look like a datafile: $datafile"
        return $EC_FALSE
    fi

    # XXX For now we just hard-code the known length of $DATAFILE_SUFFIX.
    local fullid=${name:0:-5}
    if [[ $fullid == "" || $fullid == "-"* || $fullid == *"-" ]]; then
        print-err "does not look like a datafile: $datafile"
        return $EC_FALSE
    fi

    echo $fullid
    return $EC_TRUE
}

function extract-datafile-tool() {
    local fullid=$(extract-datafile-fullid $1)
    if [ $? -ne $EC_TRUE ]; then
        return $EC_FALSE
    fi
    echo ${fullid%%-*}
    return $EC_TRUE
}

function extract-datafile-profid() {
    local fullid=$(extract-datafile-fullid $1)
    if [ $? -ne $EC_TRUE ]; then
        return $EC_FALSE
    fi
    echo ${fullid#*-}
    return $EC_TRUE
}

function get-flamegraph-file() {
    local fullid=$1
    local timestamp=$2

    if [[ "$fullid" == *"$DATAFILE_SUFFIX" ]]; then
        fullid=$(extract-datafile-fullid $fullid)
        if [ $? -ne $EC_TRUE ]; then
            return $EC_FALSE
        fi
    fi

    if [ "$timestamp" == 'yes' ]; then
        timestamp="-$(get-timestamp)"
    elif [ -z "$timestamp" ]; then
        timestamp=""
    elif [[ $timestamp != '-'* ]]; then
        timestamp="-$timestamp"
    fi

    local datadir=$(dirname $datafile)
    if [ -z "$datadir" ]; then
        # XXX Default to $PERF_DIR?
        datadir=$(pwd)
    fi
    echo "$datadir/flamegraph-$fullid$timestamp.svg"
    return $EC_TRUE
}


#######################################
# the tools

function run-perf() {
    local datafile=$1
    local frequency=$2
    local cmd=$3

    if [ -z "$frequency" ]; then
        frequency=$PERF_FREQUENCY
    fi
    if [ -z "$datafile" ]; then
        datafile=$(get-data-loc 'perf' '' "$frequency")
    elif [[ $datafile == "-"* ]]; then
        local tags=$datafile
        if [[ $tags == *".data" ]]; then
            tags="${tags:0:-5}"
        fi
        datafile=$(get-data-loc 'perf' "$tags" "$frequency")
    fi

    # We use globals since the trap execute in the context of the caller.
    __frequency_orig__=$(cat /proc/sys/kernel/perf_event_max_sample_rate)
    __maxstack_orig__=$(cat /proc/sys/kernel/perf_event_max_stack)
    #sysctl kernel.perf_event_max_sample_rate
    #sysctl kernel.perf_event_max_stack
    trap '
        # This will run at the end of the function.
        (
        set -x
        sudo sysctl -w kernel.perf_event_max_sample_rate=$__frequency_orig__
        sudo sysctl -w kernel.perf_event_max_stack=$__maxstack_orig__
        )
        trap - RETURN
    ' RETURN
    (
    set -x
    sudo sysctl -w kernel.perf_event_max_sample_rate=$frequency
    sudo sysctl -w kernel.perf_event_max_stack=$MAX_STACK
    sudo perf record \
        --all-cpus \
        -F max \
        --call-graph fp \
        --output $datafile \
        -- $cmd
        # XXX
        #--exclude-perf
    )
    local rc=$?
    if [ $? -ne $EC_TRUE ]; then
        return $EC_FALSE
    fi
    (
    set -x
    sudo chown $USER:$USER "$datafile"
    )
    return $EC_TRUE
}

function run-uftrace() {
    local datadir=$1
    local cmd=$2

    if [ -z "$datadir" ]; then
        datadir=$(get-data-loc 'uftrace' '' "$frequency")
    elif [[ $datadir == "-"* ]]; then
        local tags=$datadir
        if [[ $tags == *".data" ]]; then
            tags="${tags:0:-5}"
        fi
        datadir=$(get-data-loc 'uftrace' "$tags" "$frequency")
    fi

    (
    set -x
    uftrace record \
        --data $datadir \
        --max-stack $MAX_STACK \
        $cmd
        #--nest-libcall
    )
    if [ $? -ne $EC_TRUE ]; then
        return $EC_FALSE
    fi
    return $EC_TRUE
}

function create-flamegraph() {
    local datafile=$1
    local outfile=$2

    local fullid=$(extract-datafile-fullid $datafile)
    if [ $? -ne $EC_TRUE ]; then
        return $EC_FALSE
    fi

    if [ -z "$outfile" ]; then
        outfile=$(get-flamegraph-file $fullid)
    fi

    local tool=${fullid%%-*}  # See extract-datafile-tool.
    if [ "$tool" == 'perf' ]; then
        local scriptfile="$PERF_DIR/$fullid.script"
        local foldedfile="$PERF_DIR/$fullid.folded"
        (
        set -x
        perf script \
            --max-stack $MAX_STACK \
            --input $datafile \
            > $scriptfile
        )
        local rc=$?
        if [ $? -ne $EC_TRUE ]; then
            return $EC_FALSE
        fi
        (
        set -x
        $FLAMEGRAPH_REPO/stackcollapse-perf.pl $scriptfile > $foldedfile
        $FLAMEGRAPH_REPO/flamegraph.pl $foldedfile > $outfile
        )
    elif [ "$tool" == 'uftrace' ]; then
        (
        set -x
        uftrace dump \
            --data $datafile \
            --flame-graph \
            | $FLAMEGRAPH_REPO/flamegraph.pl \
            > $outfile
            #--sample-time 1us
        )
        if [ $? -ne $EC_TRUE ]; then
            return $EC_FALSE
        fi
    else
        fail "unsupported tool '$tool'"
    fi

    return $EC_TRUE
}


#######################################
# high-level commands

function do-flamegraph-command() {
    local tool=$1
    local tags=$2
    local cmd=$3
    local upload=$4
    local extra_arg=$5

    echo
    echo "==== warming up the disk cache ===="
    echo
    local i
    for i in {1..4}; do
        (
        set -x
        $cmd
        )
    done

    echo
    echo "==== generating profile data with $tool ===="
    echo
    local datafile_freq=$NO_FREQUENCY
    if [ -z "$extra_arg" ]; then
        if [ "$tool" == 'perf' ]; then
            extra_arg=$PERF_FREQUENCY
            datafile_freq=$extra_arg
        fi
    fi
    local datafile=$(get-data-loc "$tool" "$tags" $datafile_freq)
    if [ "$tool" == 'perf' ]; then
        local frequency=$extra_arg
        if ! run-perf "$datafile" "$frequency" "$cmd"; then
            return $EC_FALSE
        fi
    elif [ "$tool" == 'uftrace' ]; then
        if ! run-uftrace "$datafile" "$cmd"; then
            return $EC_FALSE
        fi
    else
        fail "unsupported tool '$tool'"
    fi
    echo
    echo "# data file(s): $datafile"

    echo
    echo "==== generating the flamegraph ===="
    echo
    flamegraphfile=$(get-flamegraph-file $datafile)
    if ! create-flamegraph "$datafile" "$flamegraphfile"; then
        fail "could not create flamegraph at $flamegraphfile"
    fi
    echo
    echo "# flamegraph file: $flamegraphfile"

    if [ $upload == 'yes' ]; then
        echo
        echo "==== uploading the flamegraph SVG file ===="
        echo
        #remotefile=$(basename $(get-flamegraph-file "$datafile" 'yes'))
        remotename=$(basename $flamegraphfile)
        remotefile="flamegraphs/${remotename#*-}"
        if ! upload-file $flamegraphfile $remotefile 'Add a flame graph SVG file.'; then
            print-err "upload failed!"
            return $EC_FALSE
        fi
        echo
        echo "# uploaded to $(get-upload-url $remotefile)"
    fi

    return $EC_TRUE
}


#######################################
# the script

if [ "$0" == "$BASH_SOURCE" ]; then
    # Parse the CLI args.
    tool=
    tags=
    cmd=
    upload='no'
    datafileonly='no'
    perf_frequency=
    if [ $# -gt 0 ]; then
        tool=$1
        if [[ $tool == "-"* ]]; then
            tool='perf'
            # Leave the flag in $@.
        elif [ -z "$tool" ]; then
            tool='perf'
            shift
        else
            shift
        fi
    fi
    while test $# -gt 0 ; do
        arg=$1
        shift
        case $arg in
          --tags)
            tags=$1
            shift
            if [ -n "$tags" ]; then
                if [[ $tags != '-'* ]]; then
                    tags="-$tags"
                fi
            fi
            ;;
          --datafile-only)
            datafileonly='yes'
            ;;
          --frequency)
            if [ "$tool" == 'perf' ]; then
                perf_frequency=$1
                tags="-freq$perf_frequency$tags"
                shift
            else
                fail "unsupported flag $arg"
            fi
            ;;
          --upload)
            upload='yes'
            ;;
          -*)
            fail "unsupported flag $arg"
            ;;
          *)
            cmd="$arg $@"
            break
            ;;
        esac
    done
    if [ -z "$cmd" ]; then
        fail "missing cmd"
    fi

    if [ $datafileonly == 'yes' ]; then
        datafile_freq=$NO_FREQUENCY
        if [ -z "$frequency" ]; then
            if [ "$tool" == 'perf' ]; then
                datafile_freq=$PERF_FREQUENCY
            fi
        fi
        get-data-loc "$tool" "$tags" $datafile_freq
        exit 0
    fi

    # Run the profiler and generate a flamegraph.
    do-flamegraph-command "$tool" "$tags" "$cmd" "$upload" "$perf_frequency"
fi
