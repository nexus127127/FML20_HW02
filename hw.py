
import os

# os.system('libsvm/svm-train')

os.system('libsvm/svm-train -t 1 -d 1 -c 1 HwData/Output/Scaled/train.scaled.txt HwData/Output/Model/model.txt > HwData/Output/Model/model.log.txt')


os.system('libsvm/svm-predict HwData/Output/Scaled/test.scaled.txt HwData/Output/Model/model.txt HwData/Output/Prediction/pred.txt > HwData/Output/Prediction/pred.log.txt')


pass