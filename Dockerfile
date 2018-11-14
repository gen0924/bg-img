FROM ubuntu:16.04

MAINTAINER liugen

RUN apt-get update && \
    apt-get -y install software-properties-common && \
    add-apt-repository -y ppa:jonathonf/python-3.6 && \
    apt-get update && \
    apt-get -y install python3.6 libpython3.6 python3-pip && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.5 2 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1 && \
    rm /usr/bin/python3 && \
    ln -s python3.6 /usr/bin/python3

ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && \
    echo $TZ > /etc/timezone

COPY . usr/local/src/

WORKDIR usr/local/src/

RUN pip3 install --upgrade setuptools pip && \
    pip install --no-cache-dir -r requirements.txt \
&& apt-get clean \
&& apt-get autoremove \
&& rm -rf /var/lib/apt/lists/*
CMD ["gunicorn", "-c", "gunicorn.conf", "--timeout", "10000", "main:app"]
