import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.storage.StorageLevel;
import org.deeplearning4j.api.storage.StatsStorageRouter;
import org.deeplearning4j.api.storage.impl.RemoteUIStatsStorageRouter;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.spark.api.RDDTrainingApproach;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.LoggerFactory;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;


public class CnnOnSparkCPU {

    private static final org.slf4j.Logger log = LoggerFactory.getLogger(CnnOnSparkCPU.class);
    @Parameter(names = "-batchSize", description = "Number of examples to fit each worker with")
    private static int batchSize = 20;
    @Parameter(names = "-epochs", description = "Number of epochs for training")
    private int epochs = 1;
    @Parameter(names = "-averagingFrequency", description = "averaging Frequency for Training Master to update parameters")
    private int averagingFrequency = 3;
    @Parameter(names = "-prefetchNumBatches", description = "prefetch number of Batches per worker")
    private int prefetchNumBatches = 5;
    private JavaSparkContext sparkContext;
    private TrainingMaster trainingMaster;

    public static void main(String[] args) throws IOException {

        new CnnOnSparkCPU().startUp(args);
    }

    private void startUp(String[] args) throws IOException {
        //parse the command arguments
        log.info("Dl4j===> Parsing JCommander");
        JCommander jCommander = new JCommander(this);
        try {
            jCommander.parse(args);
        } catch (ParameterException e2) {
            jCommander.usage();
            try {
                //hold this thread until J command thread display the usage
                Thread.sleep(1000);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
        //configure spark
        SparkConf sparkConf = new SparkConf();
        sparkConf.setAppName("Running CNN on Spark 2");
        sparkConf.setMaster("spark://172.17.230.254:7077");
        sparkConf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer");
        sparkConf.set("spark.kryo.registrator", "org.nd4j.Nd4jRegistrator");
        sparkConf.set("spark.kryo.insafe", "true");
        sparkConf.set("spark.kryoserializer.buffer.max", "512m");
        sparkConf.set("spark.io.compression.codec", "lzf");

        log.info("Dl4j===> ****Spark is configured, spark****");
        //create spark context
        sparkContext = new JavaSparkContext(sparkConf);
        log.info("Dl4j===> ****Spark context is created****");
        //get CNN configurations
        MultiLayerConfiguration cnnConf = new CnnConfiguration(Hyperparameters.depth,
                Hyperparameters.width, Hyperparameters.height, Hyperparameters.outputs, Hyperparameters.seed).getConf();
        log.info("Dl4j===> ****Neural Network is configured****");

        //create Spark training master
        trainingMaster = new ParameterAveragingTrainingMaster.Builder(batchSize)
                .averagingFrequency(averagingFrequency)
                .workerPrefetchNumBatches(prefetchNumBatches)
                .batchSizePerWorker(batchSize)
                .rddTrainingApproach(RDDTrainingApproach.Direct)
                .storageLevel(StorageLevel.MEMORY_AND_DISK_SER())
                .saveUpdater(true)
                .build();
        log.info("Dl4j===> ******Parameter Averaging Training Master  is created******");
        //create spark network
        SparkDl4jMultiLayer sparkDl4jMultiLayer = new SparkDl4jMultiLayer(sparkContext, cnnConf, trainingMaster);
        UIServer uiServer = UIServer.getInstance();
        uiServer.enableRemoteListener();
        StatsStorageRouter remote = new RemoteUIStatsStorageRouter("http://172.17.230.254:9001");
        sparkDl4jMultiLayer.setListeners(remote, Collections.singletonList(new StatsListener(null)));
        //sparkDl4jMultiLayer.setListeners( new ScoreIterationListener(1));
        log.info("Dl4j===> ******Spark Dl4j MultiLayer  is created******");
        log.info("Dl4j===> **********start training the Network********");
        PreprocessingData preprocessingData = new PreprocessingData();
        trainAndEvaluate(sparkDl4jMultiLayer, preprocessingData);
        //delete temporary files
        trainingMaster.deleteTempFiles(sparkContext);
        sparkContext.close();
        System.exit(0);
    }

    private void trainAndEvaluate(SparkDl4jMultiLayer sparkDl4jMultiLayer,
                                  PreprocessingData preprocessingData) throws IOException {
        log.info("Dl4j===> loading RDDs ......");
        long start = System.currentTimeMillis();
        JavaRDD<DataSet> trainData = sparkContext.parallelize(getTrainData(preprocessingData));
        long[] timeEpoches = new long[epochs];
        log.info("Dl4j===> ***********training started**********");
        for (int epo = 0; epo < epochs; epo++) {
            long startEpoch = System.currentTimeMillis();
            sparkDl4jMultiLayer.fit(trainData);
            long finishEpoch = System.currentTimeMillis();
            timeEpoches[epo] = finishEpoch - startEpoch;
            log.info("\nDl4j===> ***************************************");
            log.info("\nDl4j===> ***********Completed Epoch {}**********", epo);
            log.info("\nDl4j===> ***************************************");
        }
        long end = System.currentTimeMillis();
        log.info("Dl4j===> ***********training finished, time ={} minutes **********", ((end - start) * 1.6666666666667e-5));
        log.info("Dl4j===> ***********Start Evaluation**********");
        JavaRDD<DataSet> testData = sparkContext.parallelize(getTestData(preprocessingData));
        Evaluation evaluation = sparkDl4jMultiLayer
                .doEvaluation(testData, batchSize, new Evaluation(Hyperparameters.outputs))[0];
        long  allEnd = System.currentTimeMillis();
        log.info("Dl4j===> *************End evaluation***********");
        log.info(evaluation.stats());
        log.info("\n\nDl4j===> ***************************************");
        log.info("Dl4j===> ****finished, total time = {} minutes***", ((allEnd - start) * 1.6666666666667e-5));
        for (int epo = 0; epo < epochs; epo++) {
            log.info("Dl4j===> Epoch_{} time = {} minutes",epo , (timeEpoches[epo] * 1.6666666666667e-5));
        }
        log.info("\n\nDl4j===>***************************************");
    }


    private List<DataSet> getTrainData(PreprocessingData preprocessingData) throws IOException {
        DataSetIterator trainDataIterator = preprocessingData.getTrainDataSetIterator();
        List<DataSet> trainDataSets = new ArrayList<>();
        while (trainDataIterator.hasNext())
            trainDataSets.add(trainDataIterator.next());
        return trainDataSets;
    }

    private List<DataSet> getTestData(PreprocessingData preprocessingData) throws IOException {
        DataSetIterator testDataIterator = preprocessingData.getTestDataSetIterator();
        List<DataSet> testDataSets = new ArrayList<>();
        while (testDataIterator.hasNext())
            testDataSets.add(testDataIterator.next());
        return testDataSets;
    }
}
