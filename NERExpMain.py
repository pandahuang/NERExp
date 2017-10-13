# -*- coding:utf-8 -*-
from sys import argv
from Data.DataManager import DataManager
from Scripts.ScriptToolkit import ScriptToolkit
from Preprocessing.DataReader import DataReader
from Preprocessing import ProcessorFactory
from Model.ConditionalRandomField import CRF

if __name__ == '__main__':
    # create data manager
    DM = DataManager()
    DM.change_pwd()
    DM.source_data_file = argv[argv.index("-source") + 1] if "-source" in argv else 'CorpusLabelData_MergedFilter.txt'
    DM.remove(DM.log_wrong_sentences)

    # create datums
    DR = DataReader(source_data_file=DM.source_data_file)
    DR.standard_read()

    # create toolkits
    ST = ScriptToolkit(DM)
    features = ScriptToolkit.get_demo_features()

    # analysis
    sent_accuracys, train_times, test_times = [], [], []
    cycle_times = int(argv[argv.index("-iter") + 1]) if "-iter" in argv else 10
    for i in range(cycle_times):
        print "This is the %d-th experiments." % (i + 1)

        # data preprocessing
        crf_processor = ProcessorFactory.CRFProcessorFactory().produce(source_data_file=DM.source_data_file,
                                                                       train_file=DM.train_file, test_file=DM.test_file)
        crf_processor.get_train_data(DR.Datums)

        # training and testing
        crf_test = CRF(path_to_jar=DM.path_to_jar, prop_file=DM.prop_file, model_file=DM.model_file,
                       source_data_file=DM.source_data_file, train_file=DM.train_file, test_file=DM.test_file,
                       result_file=DM.result_file)
        crf_test.feature_config(features=features)
        sout_train, serr_train, sout_test, serr_test = crf_test.train_and_verify()
        train_datasize, train_time = ScriptToolkit.ParseTrainSoutAndSerr(sout_train, serr_train)
        sent_accuracy, test_datasize, test_time, detail_result = ScriptToolkit.ParseTestSoutAndSerr(sout_test, serr_test)
        ScriptToolkit.LogResult(sent_accuracy, DM.source_data_file, train_datasize, train_time, test_datasize, test_time)
        ScriptToolkit(DM).LogResultsAndWrongAnswer(sout_test, serr_test, detail_result)
        sent_accuracys.append(sent_accuracy)
        train_times.append(train_time)
        test_times.append(test_time)

    # result display
    print "--------------------Final Average------------------------"
    print 'Average sent_accuracy is : %f' % (sum(sent_accuracys) / cycle_times)
    print 'Average training time is : %f' % (sum([float(train_time) for train_time in train_times]) / cycle_times)
    print 'Average testing time is : %f' % (sum([float(test_time) for test_time in test_times]) / cycle_times)
