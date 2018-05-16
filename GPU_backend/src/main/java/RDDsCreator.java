import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;


public class RDDsCreator {
    public static void main(String[] args) throws IOException {
        //create an instance of DataPipeline
        PreprocessingData pipeline  = new PreprocessingData();
        DataSetIterator trainDataSetIterator = pipeline.getTrainDataSetIterator();
        //create a list of Datasets
        List<DataSet> trainDataSets = new ArrayList<>();
        while (trainDataSetIterator.hasNext())
            trainDataSets.add(trainDataSetIterator.next());
        //create an instance of spark configuration
        SparkConf sparkConf = new SparkConf();
        //set master to local and use all cores for execution
        sparkConf.setMaster("local[14]");
        sparkConf.set("spark.executor.heartbeatInterval", "72000");
        sparkConf.set("spark.network.timeout", "73000");
        sparkConf.setAppName("create RDDs");
        //create JavaSparkContext
        JavaSparkContext sparkContext = new JavaSparkContext(sparkConf);
        // create RDD for training data
        JavaRDD<DataSet> train = sparkContext.parallelize(trainDataSets,14);
        //save training RDD to specified directory
        train.saveAsObjectFile("/home/bcri/train_data2/twoClassRDDs/train");
        DataSetIterator testDataSetIterator = pipeline.getTestDataSetIterator();
        List<DataSet> testDataSets = new ArrayList<>();
        while (testDataSetIterator.hasNext())
            testDataSets.add(testDataSetIterator.next());
        //create testing RDD
        JavaRDD<DataSet> testData = sparkContext.parallelize(testDataSets, 14);
        testData.saveAsObjectFile("/home/bcri/train_data2/twoClassRDDs/test");

    }
}
