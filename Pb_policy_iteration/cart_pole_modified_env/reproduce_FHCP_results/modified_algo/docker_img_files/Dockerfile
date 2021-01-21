FROM ubuntu:latest

WORKDIR /usr/src/exp

COPY . ./

RUN apt-get update -y \
    && apt-get install -y python3-pip \
    && apt-get install -y python3-dev \
    && pip3 install --upgrade pip

RUN pip3 install --no-cache-dir -r requirements.txt \
    &&  pip3 install -e custom_cartpole\