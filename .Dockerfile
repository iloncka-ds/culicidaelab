FROM squidfunk/mkdocs-material

# required for mkdocs-git-committers-plugin-2
RUN apk add --no-cache --virtual .build-deps gcc libc-dev libxslt-dev && \
    apk add --no-cache libxslt && \
    pip install --no-cache-dir mkdocs-gallery && \
    pip install mkdocs-custom-fences && \
    pip install mkdocstrings[python] && \
    pip install mkdocs-macros-plugin && \
    apk del .build-deps

RUN pip install --no-cache-dir \
  mkdocs-git-revision-date-localized-plugin \
  # mkdocs-git-committers-plugin-2 \
  # hotfix for authors because file move resets contributors list
  git+https://github.com/tibitoth/mkdocs-git-committers-plugin-2.git@master \
  mkdocs-glightbox

RUN git config --global --add safe.directory /github/workspace

EXPOSE 8000

ENTRYPOINT ["mkdocs", "serve", "--dev-addr=0.0.0.0:8000" ]

CMD ["--config-file=mkdocs.en.yml"]
