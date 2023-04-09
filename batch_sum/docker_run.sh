# use like: $ ./docker_run.sh $(pwd)

echo $1

xhost +local:root
docker run --rm -it \
    -v $1:/examples/ \
    --gpus all \
    --user root \
    --name 'thrust_practice' \
    --env="DISPLAY" \
    thrust \
    /bin/bash -c 'cd /examples; bash'
