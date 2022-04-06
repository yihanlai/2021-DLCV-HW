#!/bin/bash
wget -O usps_extractor.pth https://www.dropbox.com/s/xosror54rhd2n1g/mnistm2usps_extractor.pth?dl=0
wget -O usps_label_predictor.pth https://www.dropbox.com/s/k5c314vqh8ukk57/mnistm2usps_label_predictor.pth?dl=0
wget -O mnistm_extractor.pth https://www.dropbox.com/s/wv0znk05oi5pwcx/svhn2mnistm_extractor.pth?dl=0
wget -O mnistm_label_predictor.pth https://www.dropbox.com/s/jvc6l07e8x5wzvb/svhn2mnistm_label_predictor.pth?dl=0
wget -O svhn_extractor.pth https://www.dropbox.com/s/enz8cf91yvurhwk/usps2svhn_extractor.pth?dl=0
wget -O svhn_label_predictor.pth https://www.dropbox.com/s/rx0q141hfehx09u/usps2svhn_label_predictor.pth?dl=0

python3 hw2_3_test.py $1 $2 $3
