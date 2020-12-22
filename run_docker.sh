#!/bin/bash


action=${1:-delete}


if [[ $action == "create" ]]; then
    docker build -t edesz/fast-api-demo .
    docker run -d --name mycontainer -p 8000:80 edesz/fast-api-demo
    docker images
    docker ps -a
# docker ps -q | xargs -L 1 docker logs -f
elif [[ $action == "delete" ]]; then
    docker stop $(docker ps -aq)
    docker rm $(docker ps -aq)
    docker rmi $(docker images -q)
fi
