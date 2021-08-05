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


FLAMEGRAPH_URL=https://github.com/brendangregg/FlameGraph

# CPython's C stacks can get pretty deep.
MAX_STACK=1000
PERF_FREQUENCY=99999  # ~100 khz


#######################################
# data files

DATAFILE_SUFFIX='.data'

NO_FREQUENCY='-'


function get-fullid() {
    local tool=$1
    local tags=$2
    local stamp=$3
    local frequency=$4

    if [ -n "$tags" ]; then
        if [ "$tags" == '-' ]; then
            tags=""
        elif [[ $tags != '-'* ]]; then
            tags="-$tags"
        fi
    fi
    if [ -z "$frequency" -o "$frequency" == '-' ]; then
        if [ "$tool" == 'perf' ]; then
            frequency=$PERF_FREQUENCY
        else
            frequency=$NO_FREQUENCY
        fi
    fi
    if [ "$frequency" != $NO_FREQUENCY ]; then
        tags="-freq$frequency$tags"
    fi

    # We add the timestamp at the end, if requested.
    if [ "$stamp" == 'yes' ]; then
        tags="$tags-$(get-timestamp)"
    fi

    echo "${tool}${tags}"
}

function resolve-data-loc() {
    local datadir=$(resolve-data-dir $1)
    local tool=$2
    local tags=$3
    local stamp=$4
    local frequency=$5

    local fullid=$(get-fullid "$tool" "$tags" "$stamp" "$frequency")
    echo "$datadir/${fullid}$DATAFILE_SUFFIX"
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

function resolve-flamegraph-file() {
    local datadir=$(resolve-data-dir $1)
    local fullid=$2

    if [[ "$fullid" == *"$DATAFILE_SUFFIX" ]]; then
        fullid=$(extract-datafile-fullid $fullid)
        if [ $? -ne $EC_TRUE ]; then
            return $EC_FALSE
        fi
    fi

    echo "$datadir/flamegraph-$fullid.svg"
    return $EC_TRUE
}


#######################################
# the tools

function run-perf() {
    local datadir=$(resolve-data-dir $1)
    local datafile=$2
    local frequency=$3
    local cmd=$4

    if [ -z "$frequency" -o "$frequency" == '-' ]; then
        frequency=$PERF_FREQUENCY
    fi
    if [ -z "$datafile" ]; then
        datafile=$(resolve-data-loc "$datadir" 'perf' '-' '-' "$frequency")
    elif [[ $datafile == "-"* ]]; then
        local tags=$datafile
        if [[ $tags == *".data" ]]; then
            tags="${tags:0:-5}"
        fi
        datafile=$(resolve-data-loc "$datadir" 'perf' "$tags" '-' "$frequency")
    fi

    ensure-dir $datadir

    # We use globals since the trap execute in the context of the caller.
    __frequency_orig__=$(cat /proc/sys/kernel/perf_event_max_sample_rate)
    __maxstack_orig__=$(cat /proc/sys/kernel/perf_event_max_stack)
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
    local datadir=$(resolve-data-dir $1)
    local uftrace_data=$2
    local cmd=$3

    if [ -z "$uftrace_data" -o "$uftrace_data" == '-' ]; then
        uftrace_data=$(resolve-data-loc "$datadir" 'uftrace' '-' '-' $NO_FREQUENCY)
    elif [[ $uftrace_data == "-"* ]]; then
        local tags=$uftrace_data
        if [[ $tags == *".data" ]]; then
            tags="${tags:0:-5}"
        fi
        uftrace_data=$(resolve-data-loc "$datadir" 'uftrace' "$tags" '-' $NO_FREQUENCY)
    fi

    ensure-dir $datadir

    (
    set -x
    uftrace record \
        --data $uftrace_data \
        --max-stack $MAX_STACK \
        $cmd
        #--nest-libcall
    )
    if [ $? -ne $EC_TRUE ]; then
        return $EC_FALSE
    fi
    return $EC_TRUE
}

function resolve-flamegraph-repo() {
    local datadir=$(resolve-data-dir $1)
    echo $datadir/FlameGraph
}

function ensure-flamegraph-repo() {
    local repodir=$1
    if [ -z "$repodir" -o "$repodir" == '-' ]; then
        repodir=$(resolve-flamegraph-repo $repodir)
    fi
    ensure-repo $FLAMEGRAPH_URL $repodir
}

function create-flamegraph() {
    local datadir=$(resolve-data-dir $1)
    local datafile=$2
    local outfile=$3

    local fullid=$(extract-datafile-fullid $datafile)
    if [ $? -ne $EC_TRUE ]; then
        return $EC_FALSE
    fi

    if [ -z "$outfile" ]; then
        outfile=$(resolve-flamegraph-file "$datadir" "$fullid")
    fi

    ensure-dir $datadir

    local tool=${fullid%%-*}
    local flamegraphrepo=$(resolve-flamegraph-repo $datadir)
    ensure-flamegraph-repo $flamegraphrepo
    if [ "$tool" == 'perf' ]; then
        local scriptfile="$datadir/$fullid.script"
        local foldedfile="$datadir/$fullid.folded"
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
        $flamegraphrepo/stackcollapse-perf.pl $scriptfile > $foldedfile
        $flamegraphrepo/flamegraph.pl $foldedfile > $outfile
        )
    elif [ "$tool" == 'uftrace' ]; then
        (
        set -x
        uftrace dump \
            --data $datafile \
            --flame-graph \
            | $flamegraphrepo/flamegraph.pl \
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
    local datadir=$1
    local tool=$2
    local tags=$3
    local stamp=$4
    local cmd=$5
    local upload=$6
    local frequency=$7  # only expected for perf

    if [ -z "$frequency" -o "$frequency" == '-' ]; then
        frequency=$NO_FREQUENCY
    elif [ "$tool" != 'perf' ]; then
        fail "got unexpected frequency"
    fi

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
echo "'$stamp'"
    local datafile=$(resolve-data-loc "$datadir" "$tool" "$tags" "$stamp" "$frequency")
    if [ "$tool" == 'perf' ]; then
        if [ "$frequency" == $NO_FREQUENCY ]; then
            frequency=$PERF_FREQUENCY
        fi
        if ! run-perf "$datadir" "$datafile" "$frequency" "$cmd"; then
            return $EC_FALSE
        fi
    elif [ "$tool" == 'uftrace' ]; then
        if ! run-uftrace "$datadir" "$datafile" "$cmd"; then
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
    flamegraphfile=$(resolve-flamegraph-file "$datadir" "$datafile")
    if ! create-flamegraph "$datadir" "$datafile" "$flamegraphfile"; then
        fail "could not create flamegraph at $flamegraphfile"
    fi
    echo
    echo "# flamegraph file: $flamegraphfile"

    if [ $upload == 'yes' ]; then
        echo
        echo "==== uploading the flamegraph SVG file ===="
        echo
        uploadname=$(basename $flamegraphfile)
        # XXX Leave the "flamegraph-" prefix?
        uploadfile="flamegraphs/${remotename#*-}"  # Drop the "flamegraph-" prefix.
        if ! upload-file "$datadir" "$flamegraphfile" "$uploadfile" 'Add a flame graph SVG file.'; then
            print-err "upload failed!"
            return $EC_FALSE
        fi
        echo
        echo "# uploaded to $(uploads-resolve-url $uploadfile)"
    fi

    return $EC_TRUE
}


#######################################
# the script

if [ "$0" == "$BASH_SOURCE" ]; then
    # Parse the CLI args.
    tool=
    cmd=
    datadir=$PERF_DIR
    tags=
    stamp='no'
    upload='no'
    datafileonly='no'
    frequency=$NO_FREQUENCY
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
        if [ "$tool" == 'perf' ]; then
            frequency=$PERF_FREQUENCY
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
                elif [ "$tags" == '-' ]; then
                    tags=""
                fi
            fi
            ;;
          --datadir)
            datadir=$1
            if [ -z "$datadir" -o "$datadir" == '-' ]; then
                datadir=$PERF_DIR
            fi
            shift
            ;;
          --datafile-only)
            datafileonly='yes'
            ;;
          --frequency)
            if [ "$tool" == 'perf' ]; then
                frequency=$1
                shift
                if [ -z "$frequency" -o "$frequency" == '-' ]; then
                    frequency=$PERF_FREQUENCY
                fi
            else
                fail "unsupported flag $arg"
            fi
            ;;
          --stamp)
            stamp='yes'
            ;;
          --no-stamp)
            stamp='no'
            ;;
          --upload)
            upload='yes'
            ;;
          --no-upload)
            upload='no'
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
    if [ -z "$tool" -o "$tool" == '-' ]; then
        fail "missing tool"
    fi
    if [ -z "$cmd" ]; then
        fail "missing cmd"
    fi

    if [ -z "$tags" ]; then
        tags='-'
    fi

    if [ $datafileonly == 'yes' ]; then
        resolve-data-loc "$datadir" "$tool" "$tags" "$stamp" "$frequency"
        exit 0
    fi

    # Run the profiler and generate a flamegraph.
    do-flamegraph-command "$datadir" "$tool" "$tags" "$stamp" "$cmd" "$upload" "$frequency"
fi
