#!/bin/sh
echo Building Triton Marian backend ...

docker build -t triton-marian-builder-only:debug .

echo Copying artifacts ...

docker container create --name extract triton-marian-builder-only:debug
docker container cp extract:/opt/tritonserver/marian_backend/build/libtriton_marian.so ./libtriton_marian.latest-debug.so
docker container rm -f extract
