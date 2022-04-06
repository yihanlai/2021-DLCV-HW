#!/bin/bash
wget -O G_model_best_ACGAN.pth https://www.dropbox.com/s/rtdzfncqcucoqy0/G_model_best_ACGAN.pth?dl=0
python3 hw2_2_test.py $1
