#!/bin/sh
set -ex

ruff check ${PACKAGE} docs dev --fix
ruff format ${PACKAGE} docs dev
