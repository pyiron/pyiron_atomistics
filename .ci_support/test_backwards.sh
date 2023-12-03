#!/bin/bash

pip install --no-deps pyiron==0.2.16
echo "Before save";
for t in tests/backwards/*save.py; do
    echo "Running $t";
    python $t
done

pip install versioneer[toml]==0.29
pip install . --no-deps --no-build-isolation
i=0;
echo "Before loading";
for t in tests/backwards/*load.py; do
    echo "Running $t";
    python $t || i=$((i+1));
done

# push error to next level
if [ "$i" -gt 0 ]; then
    exit 1;
fi;
