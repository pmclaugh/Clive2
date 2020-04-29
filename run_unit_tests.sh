#!/usr/bin/env bash
SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"
export PYTHONPATH=${SCRIPT_DIR}/src:${SCRIPT_DIR}/test/unit:${SCRIPT_DIR}/test/common

optspec=":hv-:"
inc_coverage=false
verbosity_args="-v"

pytest $verbosity_args --disable-pytest-warnings test/unit
