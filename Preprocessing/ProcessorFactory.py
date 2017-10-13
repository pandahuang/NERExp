from Preprocessing import Preprocessor
from Data.DataManager import DataManager


class Provider(object):
    def produce(self, **kw):
        pass


class CRFProcessorFactory(Provider):
    def produce(self, **kw):
        return Preprocessor.CRFPreprocessor(**kw)


class LSTMProcessorFactory(Provider):
    def produce(self, **kw):
        return Preprocessor.LSTMPreprocessor(**kw)


if __name__ == '__main__':
    DM = DataManager()
    DM.change_pwd()
    DM.source_data_file = 'CorpusLabelData_MergedFilter_Update.txt'
    crf_factory = CRFProcessorFactory()
    crf_processor = crf_factory.produce(source_data_file=DM.source_data_file, train_file=DM.train_file,
                                        test_file=DM.test_file)
    crf_processor.preprocess()
    crf_processor.get_train_data()
