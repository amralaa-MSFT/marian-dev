#!/bin/sh

set -xv
set -eu

echo Building Triton Marian backend ...

image_name='triton-marian-builder:20.09'
docker build -t "$image_name" --build-arg CACHEBUST="$(date --utc +%s)" .

echo Copying artifacts ...

docker container create --name extract "$image_name"
docker container cp extract:/opt/tritonserver/marian_backend/build/libtriton_marian.so .
docker container rm -f extract
