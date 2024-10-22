#!/usr/bin/env bash

set -ex

ruff check ${PACKAGE} docs dev
ruff format ${PACKAGE} docs dev --check
