#!/bin/sh

set -xv
set -eu

tag='0825-benchmark-2'
echo Building Triton Marian backend ...

docker build -t triton-marian-build:"$tag" .

echo Copying artifacts ...

docker container create --name extract triton-marian-build:"$tag"
docker container cp extract:/opt/tritonserver/marian_backend/build/libtriton_marian.so ./libtriton_marian-"$tag".so
docker container rm -f extract
