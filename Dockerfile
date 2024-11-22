FROM harbor.maxiv.lu.se/daq/conda-build:latest AS build

COPY conda-env.yaml /tmp/env.yaml

RUN mamba env create -f /tmp/env.yaml  && \
    conda-pack -n pipeline -o /tmp/env.tar && \
    mkdir /venv && cd /venv && tar xf /tmp/env.tar && \
    rm /tmp/env.tar && \
    /venv/bin/conda-unpack

FROM harbor.maxiv.lu.se/dockerhub/library/ubuntu:latest AS runtime
ENV PATH /venv/bin:$PATH
COPY --from=build /venv /venv

ENV HDF5_PLUGIN_PATH /venv/lib/hdf5/plugin

RUN apt-get update && apt-get install -y build-essential

ARG CI_COMMIT_SHA=0000
ARG CI_COMMIT_REF_NAME=none
ARG CI_COMMIT_TIMESTAMP=0
ARG CI_PROJECT_URL=none

WORKDIR /tmp

COPY requirements.txt /tmp/requirements.txt

RUN python -m pip --no-cache-dir install -r requirements.txt

COPY src /tmp/src
COPY <<EOF /etc/build_git_meta.json
{
"commit_hash": "${CI_COMMIT_SHA}",
"branch_name": "${CI_COMMIT_REF_NAME}",
"timestamp": "${CI_COMMIT_TIMESTAMP}",
"repository_url": "${CI_PROJECT_URL}"
}
EOF

CMD ["dranspose"]
