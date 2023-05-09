FROM mambaorg/micromamba:1.4.2

COPY --chown=$MAMBA_USER:$MAMBA_USER env.yaml /tmp/env.yaml

RUN micromamba install --yes --file /tmp/env.yaml


ARG MAMBA_DOCKERFILE_ACTIVATE=1

USER root
RUN adduser --disabled-password --gecos "" cbrc

USER cbrc

COPY --chown=cbrc:cbrc segmentation /cbrc/segmentation/

COPY --chown=cbrc:cbrc models /cbrc/models/

WORKDIR /cbrc/segmentation
