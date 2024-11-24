FROM python:3.10

WORKDIR /home/app
ENV PYTHONPATH=/home/app

COPY requirements.txt /dependencies/requirements.txt
RUN pip install -r /dependencies/requirements.txt

COPY . /home/app