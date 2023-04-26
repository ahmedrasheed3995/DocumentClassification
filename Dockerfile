FROM python:3.8

RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

COPY ./requirements.txt /requirements.txt

WORKDIR /

RUN pip install -r requirements.txt

COPY inference.py /
COPY models /models

ENV PYHTONUNBUFFERED=1
RUN apt-get update \
  && apt-get -y install tesseract-ocr

ENTRYPOINT [ "python" ]
#CMD ["inference.py" ]
