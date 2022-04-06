#!/bin/bash
wget -O model_best_vgg16_with_bn.pth https://www.dropbox.com/s/dqwcmuh3khgx47g/model_best_vgg16_with_bn.pth?dl=1
python3 hw1_1_test.py $1 $2 
