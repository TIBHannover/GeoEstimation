FROM tensorflow/tensorflow:1.15.2-gpu-py3
MAINTAINER TIB-Visual-Analytics

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install -y caffe-cpu=1.0.0-6
RUN pip install matplotlib==3.1.3
RUN pip install imageio==2.6.1
RUN pip install numpy==1.15.4
RUN pip install s2sphere==0.2.5
