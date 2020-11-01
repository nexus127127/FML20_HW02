import os
import random
import copy
import statistics
import matplotlib.pyplot as plt


def PrepareFiles():

    # Using readlines() 
    file_original = open("Abalone/abalone.data", "r")
    Lines = file_original.readlines()

    # list_lines = []
    list_dataset = []
    # Strips the newline character 
    for line in Lines:
        list_data = []
        line = line.replace('\n','')
        line_split = line.split(',')

        label_int = int(line_split[len(line_split)-1])

        if label_int >= 1 and label_int <= 9:
            list_data.append(str(1))
        else:
            list_data.append(str(0))

        if(line_split[0]=="F"):
            list_data.append(str(1))
        elif(line_split[0]=="M"):
            list_data.append(str(2))
        elif(line_split[0]=="I"):
            list_data.append(str(3))
        
        for i in range(1,len(line_split)-1):
            val = line_split[i]
            # val_modified = " " + str(i) + ":" + val
            # line_modified = line_modified+val_modified
            list_data.append(str(val))

        # list_lines.append(line_modified)
        list_dataset.append(list_data)

    file_original.close()

    list_training, list_testing = SplitTrainAndTest(list_dataset)
    WriteDataToFiles(list_training, "HwData/unscaled.train")
    WriteDataToFiles(list_testing, "HwData/unscaled.test")


def MainCV(list_train, list_test, file_path, file_path_cv, d_fixed, k_fixed):

    list_CVset = GenerateCrossValidationDataset(list_train)
    WriteCrossValidationDataset(list_CVset, file_path_cv)

    d_error_min = 9999
    k_error_min = 9999
    error_min = 9999

    d_start = 1
    d_end = 5
    if(d_fixed >= 0):
        d_start = d_fixed
        d_end = d_fixed + 1


    k_start = -10
    k_end = 11
    if(k_fixed > -9999):
        k_start = k_fixed
        k_end = k_fixed + 1

    # list_k_range = list(range(0,11,2))
    list_k_range = list(range(-10,11))


    

    list_cv_error_rtn = []

    for d in range(d_start, d_end):
        list_k = list_k_range
        list_error = []
        list_std_m1 = []
        list_std_p1 = []
        for k in list_k_range:

            list_error_k = []
            for i_cv in range(0,10):
                c_k = 2 ** k
                path_cv = file_path_cv + str(i_cv) + '/'
                CrossValidateOneSetting(d, c_k, path_cv)
                path_cv_dev = path_cv + 'cv.dev'
                path_cv_dev_pred = path_cv + 'cv.dev.pred'
                error_mean, error_std = ValidateAccuracy(path_cv_dev, path_cv_dev_pred)

                list_error_k.append(error_mean)
                

            
            error_avg = statistics.mean(list_error_k)
            error_std = statistics.stdev(list_error_k)
            list_error.append(error_avg)
            list_std_m1.append(error_avg - error_std)
            list_std_p1.append(error_avg + error_std)

            if(error_avg < error_min):
                error_min = error_avg
                d_error_min = d
                k_error_min = k


            asdf = 0

        str_figure_d = "CV Error: d = " + str(d)
        plt.figure(str_figure_d)
        str_label_ed = "Average Error"
        line_cv_error_ed, = plt.plot(list_k, list_error, label=str_label_ed)
        str_label_m1d = "-1 Std"
        line_cv_error_m1d, = plt.plot(list_k, list_std_m1, label=str_label_m1d, linestyle='dashed')
        str_label_p1d = "+1 Std"
        line_cv_error_p1d, = plt.plot(list_k, list_std_p1, label=str_label_p1d, linestyle='dashed')
        plt.legend(handles=[line_cv_error_ed, line_cv_error_m1d, line_cv_error_p1d])

        list_cv_error_rtn = copy.deepcopy(list_error)

        aaa = asdf + 1

    

    # k_error_min = 10 # Fixed k to 10 *******************************************************
    c_k = 2 ** k_error_min
    list_d = list(range(d_start,d_end))

    list_error_cv = []
    list_error_cv_m1d = []
    list_error_cv_p1d = []
    list_nsv_cv = []
    list_nbsv_cv = []

    list_error_full = []
    list_error_full_m1d = []
    list_error_full_p1d = []
    list_nsv_full = []
    list_nbsv_full = []

    list_cv_error_fk_rtn = []
    list_test_error_fk_rtn = []

    for d in range(d_start, d_end):
        
        list_error_d = []
        std_avg = 0
        nsv_avg = 0
        nbsv_avg = 0

        for i_cv in range(0,10):
            path_cv = file_path_cv + str(i_cv) + '/'
            CrossValidateOneSetting(d, c_k, path_cv)
            path_cv_dev = path_cv + 'cv.dev'
            path_cv_dev_pred = path_cv + 'cv.dev.pred'
            error_mean, error_std = ValidateAccuracy(path_cv_dev, path_cv_dev_pred)
            list_error_d.append(error_mean)

            nsv, nbsv = LoadnSVnBSV(path_cv + 'cv.train.log')
            std_avg += error_std
            nsv_avg += nsv
            nbsv_avg += nbsv

        ValidateOnFullTrainAndTest(d, c_k, file_path)
        file_path_test = file_path + "scaled.test"
        file_path_test_pred = file_path + "scaled.test.pred"
        error_full, error_full_std = ValidateAccuracy(file_path_test, file_path_test_pred)
        file_path_model_log = file_path + "scaled.train.model.log"
        nsv_full, nbsv_full = LoadnSVnBSV(file_path_model_log)

        
        error_avg = statistics.mean(list_error_d)
        error_std = statistics.stdev(list_error_d)
        std_avg = std_avg/10
        nsv_avg = nsv_avg/10
        nbsv_avg = nbsv_avg/10

        list_error_cv.append(error_avg)
        list_error_cv_m1d.append(error_avg - error_std)
        list_error_cv_p1d.append(error_avg + error_std)
        list_nsv_cv.append(nsv_avg)
        list_nbsv_cv.append(nbsv_avg)

        list_error_full.append(error_full)
        list_error_full_m1d.append(error_full - error_full_std)
        list_error_full_p1d.append(error_full + error_full_std)
        list_nsv_full.append(nsv_full)
        list_nbsv_full.append(nbsv_full)

    
    list_cv_error_fk_rtn = copy.deepcopy(list_error_cv)
    list_test_error_fk_rtn = copy.deepcopy(list_error_full)

    str_figure_cv_test_error = "CV and Test Error: Fixed k = " + str(k_error_min)
    plt.figure(str_figure_cv_test_error)
    line_fixed_k_cv_error, = plt.plot(list_d, list_error_cv, label="CV Error")
    line_fixed_k_test_error, = plt.plot(list_d, list_error_full, label="Test Error")
    plt.legend(handles=[line_fixed_k_cv_error, line_fixed_k_test_error])

    str_figure_CV_5error = "CV Error: Fixed k = " + str(k_error_min)
    plt.figure(str_figure_CV_5error)
    line_fixed_k_cv_error_avg, = plt.plot(list_d, list_error_cv, label="Average Error")
    line_fixed_k_cv_error_m1, = plt.plot(list_d, list_error_cv_m1d, label="-1 Std", linestyle='dashed')
    line_fixed_k_cv_error_p1, = plt.plot(list_d, list_error_cv_p1d, label="+1 Std", linestyle='dashed')
    plt.legend(handles=[line_fixed_k_cv_error_avg, line_fixed_k_cv_error_m1, line_fixed_k_cv_error_p1])

    str_figure_Test_5error = "Test Error: Fixed k = " + str(k_error_min)
    plt.figure(str_figure_Test_5error)
    line_fixed_k_test_error_avg, = plt.plot(list_d, list_error_full, label="Average Error")
    line_fixed_k_test_error_m1, = plt.plot(list_d, list_error_full_m1d, label="-1 Std", linestyle='dashed')
    line_fixed_k_test_error_p1, = plt.plot(list_d, list_error_full_p1d, label="+1 Std", linestyle='dashed')
    plt.legend(handles=[line_fixed_k_test_error_avg, line_fixed_k_test_error_m1, line_fixed_k_test_error_p1])

    str_figure_5nsv = "Fixed k = " + str(k_error_min) + " Num of SV"
    plt.figure(str_figure_5nsv)
    line_fixed_k_cv_nsv, = plt.plot(list_d, list_nsv_cv, label="CV: Number of Support Vectors")
    line_fixed_k_test_nsv, = plt.plot(list_d, list_nsv_full, label="Test: Number of Support Vectors")
    plt.legend(handles=[line_fixed_k_cv_nsv, line_fixed_k_test_nsv])

    str_figure_5nbsv = "Fixed k = " + str(k_error_min) + " Num of BSV"
    plt.figure(str_figure_5nbsv)
    line_fixed_k_cv_nbsv, = plt.plot(list_d, list_nbsv_cv, label="CV: Number of Bounded Support Vectors")
    line_fixed_k_test_nbsv, = plt.plot(list_d, list_nbsv_full, label="Test: Number of Bounded Support Vectors")
    plt.legend(handles=[line_fixed_k_cv_nbsv, line_fixed_k_test_nbsv])


    
    asdf = 2

    return error_min, k_error_min, list_cv_error_rtn, list_cv_error_fk_rtn, list_test_error_fk_rtn

    # list_labels_testing, list_features_testing = SeperateLabels(list_test)
    # WriteTestingDataToFiles(list_labels_testing, list_features_testing, "HwData/a1.testlabel", "HwData/a1.testdata")

    pass

    # writing to file 
    # file_modified = open('abalone_mod_full.data', 'w')
    # file_modified.writelines(list_lines)
    # file_modified.close()

    pass

def LoadPreprocessedData(file_name):

    file_load = open(file_name, "r")
    Lines = file_load.readlines()
    list_data = []

    for line in Lines:
        line = line.replace('\n','')
        data = line.split()

        list_data.append(data)

    return list_data

def SplitTrainAndTest(list_data):

    list_training = list_data[:3133]
    list_testing = list_data[3133:]

    return list_training, list_testing


    # file_modified = open('training.data', 'w')
    # file_modified.writelines(list_training)
    # file_modified.close()

    # file_modified = open('testing.data', 'w')
    # file_modified.writelines(list_testing)
    # file_modified.close()

def GenerateCrossValidationDataset(list_train):

    list_data = copy.deepcopy(list_train)

    random.shuffle(list_data)

    list_CVset = []

    size_train = len(list_data)

    size_train_10 = int(size_train/10) #313

    idx_start = 0
    for i in range(0,9):
        set_i = list_data[idx_start:idx_start + 313]
        list_CVset.append(set_i)
        idx_start = idx_start + 313

    set_last = list_data[idx_start:]
    list_CVset.append(set_last)

    return list_CVset

def WriteCrossValidationDataset(list_CVset, folder_path):

    num_set = len(list_CVset)

    # for i in range(0, num_set):
    #     list_data = list_CVset[i]
    #     file_name_i = file_name + str(i+1)
    #     WriteDataToFiles(list_data, file_name_i)

    for i in range(0, num_set):

        list_data_train_i = []
        list_data_dev_i = []

        for j in range(0, num_set):
            
            list_data_j = list_CVset[j]

            if( i == j):
                list_data_dev_i = list_data_j
            else:
                list_data_train_i = list_data_train_i + list_data_j

        file_name_train_i = folder_path + str(i) + "/cv.train"
        file_name_dev_i = folder_path + str(i) + "/cv.dev"
        WritePreprocessedDataToFiles(list_data_train_i, file_name_train_i)
        WritePreprocessedDataToFiles(list_data_dev_i, file_name_dev_i)


    return

def CrossValidateOneSetting(d, C, file_path_cv):

    cmd_train_libsvm = 'libsvm/svm-train -t 1 '
    cmd_train_d = '-d ' + str(d) + ' '
    cmd_train_c = '-c ' + str(C) + ' '
    cmd_train_path_train = file_path_cv+'cv.train '
    cmd_train_path_model = file_path_cv+'cv.model '
    cmd_train_path_log = '> ' + file_path_cv+'cv.train.log'
    cmd_train_full = cmd_train_libsvm + cmd_train_d + cmd_train_c + cmd_train_path_train + cmd_train_path_model + cmd_train_path_log
    # os.system('libsvm/svm-train -t 1 -d 1 -c 1 HwData/Output/Scaled/train.scaled.txt HwData/Output/Model/model.txt > HwData/Output/Model/model.log.txt')
    os.system(cmd_train_full)

    cmd_test_dev_libsvm = 'libsvm/svm-predict '
    cmd_test_dev_path_dev = file_path_cv + 'cv.dev '
    cmd_test_dev_path_model = file_path_cv + 'cv.model '
    cmd_test_dev_path_pred = file_path_cv + 'cv.dev.pred '
    cmd_test_dev_path_log = '> ' + file_path_cv + 'cv.dev.pred.log'
    # os.system('libsvm/svm-predict HwData/Output/Scaled/test.scaled.txt HwData/Output/Model/model.txt HwData/Output/Prediction/pred.txt > HwData/Output/Prediction/pred.log.txt')
    cmd_test_dev_full = cmd_test_dev_libsvm + cmd_test_dev_path_dev + cmd_test_dev_path_model + cmd_test_dev_path_pred + cmd_test_dev_path_log
    os.system(cmd_test_dev_full)


    pass


def ValidateOnFullTrainAndTest(d, C, file_path):

    cmd_train_libsvm = 'libsvm/svm-train -t 1 '
    cmd_train_d = '-d ' + str(d) + ' '
    cmd_train_c = '-c ' + str(C) + ' '
    cmd_train_path_train = file_path + 'scaled.train '
    cmd_train_path_model = file_path + 'scaled.train.model '
    cmd_train_path_log = '> ' + file_path + 'scaled.train.model.log'
    cmd_train_full = cmd_train_libsvm + cmd_train_d + cmd_train_c + cmd_train_path_train + cmd_train_path_model + cmd_train_path_log
    # os.system('libsvm/svm-train -t 1 -d 1 -c 1 HwData/Output/Scaled/train.scaled.txt HwData/Output/Model/model.txt > HwData/Output/Model/model.log.txt')
    os.system(cmd_train_full)

    cmd_test_libsvm = 'libsvm/svm-predict '
    cmd_test_path_test = file_path + 'scaled.test '
    cmd_test_path_model = file_path + 'scaled.train.model '
    cmd_test_path_pred = file_path + 'scaled.test.pred '
    cmd_test_path_log = '> ' + file_path + 'scaled.test.pred.log'
    # os.system('libsvm/svm-predict HwData/Output/Scaled/test.scaled.txt HwData/Output/Model/model.txt HwData/Output/Prediction/pred.txt > HwData/Output/Prediction/pred.log.txt')
    cmd_test_full = cmd_test_libsvm + cmd_test_path_test + cmd_test_path_model + cmd_test_path_pred + cmd_test_path_log
    os.system(cmd_test_full)

def SeperateLabels(list_data):

    list_labels = []
    list_features = []

    for i in range(0, len(list_data)):
        data = list_data[i]
        label = data[0]
        features = data[1:]

        list_labels.append(label)
        list_features.append(features)

    return list_labels, list_features

def WriteDataToFiles(list_data, file_name_full):

    list_lines = []
    for i in range(0, len(list_data)):
        data = list_data[i]

        line = data[0]
        for j in range(1, len(data)):
            val = data[j]
            line += " " + str(j) + ":" + str(val)
        line += "\n"
        list_lines.append(line)

    # writing to file 
    file_modified = open(file_name_full, 'w')
    file_modified.writelines(list_lines)
    file_modified.close()

    return

def WritePreprocessedDataToFiles(list_data, file_name_full):

    list_lines = []
    for i in range(0, len(list_data)):
        data = list_data[i]

        line = ""
        for j in range(0, len(data)):
            val = data[j]
            line += str(val) + " "

        line += "\n"
        list_lines.append(line)

    # writing to file 
    file_modified = open(file_name_full, 'w')
    file_modified.writelines(list_lines)
    file_modified.close()

    return

def WriteTestingDataToFiles(list_labels, list_data, file_name_label_full, file_name_data_full):

    list_lines_labels = []
    list_lines_data = []
    for i in range(0, len(list_labels)):
        label = list_labels[i]
        list_lines_labels.append(str(label)+"\n")

        data = list_data[i]
        line = ""
        for j in range(0, len(data)):
            val = data[j]
            line += str(j+1) + ":" + str(val) + " "

        line = line[0:len(line)-1]
        line += "\n"
        list_lines_data.append(line)

    # writing to file 
    file_label= open(file_name_label_full, 'w')
    file_label.writelines(list_lines_labels)
    file_label.close()

    file_data = open(file_name_data_full, 'w')
    file_data.writelines(list_lines_data)
    file_data.close()

    return

def ValidateAccuracy(file_name_test, file_name_pred):

    file_test = open(file_name_test, "r")
    Lines_test = file_test.readlines()

    file_pred = open(file_name_pred, "r")
    Lines_pred = file_pred.readlines()

    counter = 0

    list_match = []
    
    for i in range(0,len(Lines_test)):
        label_test = Lines_test[i]
        label_test = label_test.split()
        label_test = label_test[0]

        label_pred = Lines_pred[i]
        label_pred = label_pred.replace("\n", "")
        label_pred = label_pred.replace(" ", "")

        if(label_test != label_pred):
            list_match.append(1)
        else:
            list_match.append(0)

    error_mean = statistics.mean(list_match)
    error_std = statistics.stdev(list_match)

    file_test.close()
    file_pred.close()

    return error_mean, error_std

def LoadnSVnBSV(file_name_train_log):

    file_train_log = open(file_name_train_log, "r")
    Lines = file_train_log.readlines()

    for line in Lines:
      if(line.find('nSV = ') != -1 and line.find('nBSV = ') != -1):
        line = line.replace(',', '')
        line = line.replace('\n', '')
        line_split = line.split(' ')
        str_nsv = line_split[2]
        nsv = float(str_nsv)
        str_nbsv = line_split[5]
        nbsv = float(str_nbsv)
        break


    return nsv, nbsv



if __name__ == "__main__":
    
    # PrepareFiles()
    # list_train = LoadPreprocessedData('HwData/scaled.train')
    # list_test = LoadPreprocessedData('HwData/scaled.test')
    # d_fixed = -1
    # k_fixed = -9999
    # error_min, k_error_min, list_cv_error_rtn, list_cv_error_fk_rtn, list_test_error_fk_rtn = MainCV(list_train, list_test, "HwData/", "CV/cv", d_fixed, k_fixed)

    # list_train_1 = LoadPreprocessedData('HwDataC5D/d1/scaled.train')
    # list_test_1 = LoadPreprocessedData('HwDataC5D/d1/scaled.test')
    # d_fixed_1 = 1
    # k_fixed_1 = -9999
    # error_min_1, k_error_min_1, list_cv_error_rtn_1, list_cv_error_fk_rtn_1, list_test_error_fk_rtn_1 = MainCV(list_train_1, list_test_1, "HwDataC5D/d1/", "CV/cv", d_fixed_1, k_fixed_1)

    # list_train_2 = LoadPreprocessedData('HwDataC5D/d2/scaled.train')
    # list_test_2 = LoadPreprocessedData('HwDataC5D/d2/scaled.test')
    # d_fixed_2 = 2
    # k_fixed_2 = -9999
    # error_min_2, k_error_min_2, list_cv_error_rtn_2, list_cv_error_fk_rtn_2, list_test_error_fk_rtn_2 = MainCV(list_train_2, list_test_2, "HwDataC5D/d2/", "CV/cv", d_fixed_2, k_fixed_2)

    # list_train_3 = LoadPreprocessedData('HwDataC5D/d3/scaled.train')
    # list_test_3 = LoadPreprocessedData('HwDataC5D/d3/scaled.test')
    # d_fixed_3 = 3
    # k_fixed_3 = -9999
    # error_min_3, k_error_min_3, list_cv_error_rtn_3, list_cv_error_fk_rtn_3, list_test_error_fk_rtn_3 = MainCV(list_train_3, list_test_3, "HwDataC5D/d3/", "CV/cv", d_fixed_3, k_fixed_3)

    # list_train_4 = LoadPreprocessedData('HwDataC5D/d4/scaled.train')
    # list_test_4 = LoadPreprocessedData('HwDataC5D/d4/scaled.test')
    # d_fixed_4 = 4
    # k_fixed_4 = -9999
    # error_min_4, k_error_min_4, list_cv_error_rtn_4, list_cv_error_fk_rtn_4, list_test_error_fk_rtn_4 = MainCV(list_train_4, list_test_4, "HwDataC5D/d4/", "CV/cv", d_fixed_4, k_fixed_4)

    # list_k = list(range(0,11,2))
    # str_figure_poly_cv = "Poly Kernel CV Error"
    # plt.figure(str_figure_poly_cv)
    # line_poly_cv_d1, = plt.plot(list_k, list_cv_error_rtn_1, label="d = 1")
    # line_poly_cv_d2, = plt.plot(list_k, list_cv_error_rtn_2, label="d = 2")
    # line_poly_cv_d3, = plt.plot(list_k, list_cv_error_rtn_3, label="d = 3")
    # line_poly_cv_d4, = plt.plot(list_k, list_cv_error_rtn_4, label="d = 4")
    # plt.legend(handles=[line_poly_cv_d1, line_poly_cv_d2, line_poly_cv_d3, line_poly_cv_d4])

    # list_d = list(range(1,5))
    # str_figure_poly_fk = "Poly Kernel Fixed k CV Error: k = 10"
    # plt.figure(str_figure_poly_fk)
    # list_poly_cv_error_fk = [list_cv_error_fk_rtn_1[0], list_cv_error_fk_rtn_2[0], list_cv_error_fk_rtn_3[0], list_cv_error_fk_rtn_4[0]]
    # list_poly_test_error_fk = [list_test_error_fk_rtn_1[0], list_test_error_fk_rtn_2[0], list_test_error_fk_rtn_3[0], list_test_error_fk_rtn_4[0]]
    # line_poly_cv_error_fk, = plt.plot(list_d, list_poly_cv_error_fk, label="CV Error")
    # line_poly_test_error_fk, = plt.plot(list_d, list_poly_test_error_fk, label="Test Error")
    # plt.legend(handles=[line_poly_cv_error_fk, line_poly_test_error_fk])
    

    plt.show()

    asdf = 0

    # ValidateAccuracy("HwDataC5D/scaled.test", "HwDataC5D/scaled.test.pred")
    # nsv, nbsv = LoadnSVnBSV('CV/cv0/cv.train.log')


    # ValidateOnFullTrainAndTest(1, 2**10)
    # accuracy = ValidateAccuracy("HwData/scaled.test", "HwData/scaled.test.pred")
    # nsv, nbsv = LoadnSVnBSV('HwData/scaled.train.model.log')

    asdf = 1

    # plt.figure("TestStr1")
    # plt.figure("TestStr2")
    # plt.show()

    
    pass