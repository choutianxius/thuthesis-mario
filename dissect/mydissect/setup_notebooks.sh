#!/bin/bash

# Set up symlinks
for DIRNAME in resources results netdissect experiment
do
  ln -sfn ../${DIRNAME} .
done
