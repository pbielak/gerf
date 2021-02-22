#!/bin/bash

GPU="${1}"

docker run -d \
	   --name gerf \
	   -v "${PWD}:/app" \
	   --gpus "device=${GPU}" \
	   --ipc=host \
	   gerf:latest /bin/bash -c "trap : TERM INT; sleep infinity & wait"
