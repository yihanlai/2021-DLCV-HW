#!/bin/bash
wget -O model_best_FCN8.pth https://www.dropbox.com/s/13sgjp1qfpow7yc/model_best_FCN8.pth?dl=1
python3 hw1_2_test.py $1 $2 
