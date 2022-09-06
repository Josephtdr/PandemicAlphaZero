# syntax=docker/dockerfile:1

FROM python:3.10.5

WORKDIR /learn

COPY requirements.txt requirements.txt 

RUN apt-get update
RUN apt-get install -y swig 

RUN pip3 install -r requirements.txt

COPY . . 

