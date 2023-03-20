#!/bin/bash

P=(1 2)
K=(1 3 5)
weights=('uniform' 'distance')

for p in "${P[@]}"; do
    for k in "${K[@]}"; do
        for w in "${weights[@]}"; do
            python PlantillaAComentar.py "$p" "$k" "$w"
        done
    done
done
