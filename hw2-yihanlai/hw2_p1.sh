#!/bin/bash
wget -O G_model_best.pth https://www.dropbox.com/s/slp6a8pfih7sl1h/G_model_best.pth?dl=0
python3 hw2_1_test.py $1
