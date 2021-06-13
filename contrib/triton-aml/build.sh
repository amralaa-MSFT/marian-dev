#!/bin/sh
echo Building Triton Marian backend ...

image_name='triton-marian-builder-only:21.09-debug'
docker build -t "$image_name" .

echo Copying artifacts ...

docker container create --name extract "$image_name"
docker container cp extract:/opt/tritonserver/marian_backend/build/libtriton_marian.so ./libtriton_marian.latest-debug.so
docker container rm -f extract
