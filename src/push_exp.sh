#!/bin/bash

cd ../experiments_results
git pull
git add .
git commit -m "new experiments"
git push