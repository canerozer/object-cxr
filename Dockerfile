FROM anibali/pytorch:1.5.0-cuda10.2

USER root

RUN pip install \
    jupyterlab==1.2.4 \
    matplotlib==3.1.2 \
    numpy==1.18.1 \
    pandas==0.25.3 \
    scikit-image==0.16.2 \
    scikit-learn==0.22.1 \
    pyyaml

# Create a working directory
RUN mkdir /dev-app
WORKDIR /dev-app

CMD ["python3"]