FROM ghcr.io/mamba-org/micromamba:latest AS build

COPY conda-env.yaml /tmp/env.yaml

RUN micromamba install --copy -y -n base -f /tmp/env.yaml && \
  rm -rf /opt/conda/pkgs
# Check list of packages installed
RUN micromamba list -n base

FROM harbor.maxiv.lu.se/dockerhub/library/ubuntu:latest AS runtime

COPY --from=build /opt/conda /opt/conda

ENV PATH /opt/conda/bin:$PATH
ENV HDF5_PLUGIN_PATH /opt/conda/lib/hdf5/plugin

ARG CI_COMMIT_SHA=0000
ARG CI_COMMIT_REF_NAME=none
ARG CI_COMMIT_TIMESTAMP=0
ARG CI_PROJECT_URL=none

WORKDIR /tmp

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
