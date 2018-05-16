import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import java.io.File;
import java.io.IOException;
import java.util.Random;

public class PreprocessingData {

    private Random rnd;
    private ImageRecordReader train;
    private ImageRecordReader test;
    private DataNormalization scalier;
    private  ParentPathLabelGenerator labelGenerator;
    private InputSplit trainSplit;

    public PreprocessingData()  throws IOException {
        rnd = new Random(Hyperparameters.seed);
        loadSplitData();
    }

    private void loadSplitData() throws IOException {
        File DataDirectory = new File(Hyperparameters.dataPath);
        String[] imageExtensions = NativeImageLoader.ALLOWED_FORMATS;
        labelGenerator = new ParentPathLabelGenerator();
        BalancedPathFilter balancedPathFilter = new BalancedPathFilter(rnd,
                imageExtensions, labelGenerator);
        //rnd is used when splitting dataSet into train and test
        FileSplit inputSplit = new FileSplit(DataDirectory, imageExtensions, rnd);
        InputSplit[] splitsFiles = inputSplit.sample(balancedPathFilter,
                Hyperparameters.trainSize, 100 - Hyperparameters.trainSize);
        trainSplit = splitsFiles[0];
        InputSplit testSplit = splitsFiles[1];
        test = new ImageRecordReader(Hyperparameters.height, Hyperparameters.width,
                Hyperparameters.depth, labelGenerator);
        scalier = new ImagePreProcessingScaler(0, 1);
        test.initialize(testSplit);
    }

    public DataSetIterator getTestDataSetIterator(){
        DataSetIterator testDataSetIterator = new  RecordReaderDataSetIterator(test, Hyperparameters.minBatchSize,
                Hyperparameters.labelIndex, Hyperparameters.outputs);
        testDataSetIterator.setPreProcessor(scalier);
        return testDataSetIterator;
    }

    public DataSetIterator getTrainDataSetIterator() throws IOException {
        train = new ImageRecordReader(Hyperparameters.height, Hyperparameters.width,
                Hyperparameters.depth, labelGenerator);
        train.initialize(trainSplit);
        DataSetIterator trainDataSetIterator = new  RecordReaderDataSetIterator(train,
                Hyperparameters.minBatchSize, Hyperparameters.labelIndex, Hyperparameters.outputs);
        scalier.fit(trainDataSetIterator);
        trainDataSetIterator.setPreProcessor(scalier);
        return trainDataSetIterator;
    }
}
