#!/bin/bash

op=$(basename $(pwd))

run_install() {
    pushd $op
    source build.sh
    if [ $? -eq 0 ]; then
        ./custom_opp_openEuler_aarch64.run
    else
        echo "build error!"
    fi
    popd
}

run_test() {
    local op=$(echo "$op" | sed -e "s/\b\(.\)/\u\1/")
    find "tests" -maxdepth 1 -type d ! -path "tests" | while read -r case; do
        echo "=> running test: $(basename $case)"
        pushd $case > /dev/null
        source run.sh
        popd > /dev/null
    done
}

case $1 in
    install)
        run_install
        ;;
    test)
        run_test
        ;;
    *)
        echo "Usage: $0 {install|test}"
        exit 1
        ;;
esac