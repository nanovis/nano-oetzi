#!/bin/bash

set -euxo pipefail


mkdir -p data

wget -nc -P data/ https://github.com/nanovis/nano-oetzi-webgpu/raw/main/data/ts_16_256/ts_16_bin4-256x256.json
wget -nc -P data/ https://github.com/nanovis/nano-oetzi-webgpu/raw/main/data/ts_16_256/ts_16_bin4-256x256.raw


chmod -R a+rx data/
