#!/usr/bin/env bash

function add-user() {
    local user=$1

    local home="/home/$user"

    (
    set -x
    sudo adduser --gecos '' --disabled-password $user
    sudo usermod -a -G sudo $user
    #sudo vigr
    #sudo vigr -s
    sudo passwd $user
    
    sudo passwd --status $user
    )
}

function set-up-user() {
    local user=$1
    local ghuser=$2
    if [ -z "$ghuser" ]; then
        ghuser=$user
    fi

    local home="/home/$user"
    local bashrc="$home/.bashrc"
    local bash_common="source ~benchmarking/BENCH/.bashrc-common.sh"

    (
    set -x
    #sudo --login --user $user bash
    sudo --login --user $user ssh-import-id gh:$ghuser
    #sudo bash -c "echo >> $bashrc"
    sudo bash -c "printf "'"'"\n$bash_common\n"'"'" >> $bashrc"
    )

    echo "# bench-ssh ssh-import-id gh:$ghuser"
    (
    set -x
    sudo --login --user benchmarking SSH_AUTH_SOCK="$SSH_AUTH_SOCK" ssh -A -p 22222 benchmarking@localhost ssh-import-id gh:$ghuser
    )
}

function remove-user() {
    local user=$1

    (
    set -x
    #deluser $user sudo
    deluser --backup --remove-home $user
    )
}


##################################
# the script

user=$1
ghuser=$user
if [ -z "$user" ]; then
    >&2 echo "ERROR: missing user arg"
    exit 1
fi

add-user $user
set-up-user $user $ghuser
