#!/usr/bin/env bash
set -eux
docker run --rm -v $PWD:/workspace -w /workspace python:3.6-buster ./create_lockfile.sh
CIRCLE_CI_LOCK=./poetry.circleci.lock
cp ./poetry.lock $CIRCLE_CI_LOCK

echo "Updated $CIRCLE_CI_LOCK"
