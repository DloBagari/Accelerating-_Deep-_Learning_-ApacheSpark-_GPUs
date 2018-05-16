interface Hyperparameters {
    int width = 32;
    int height = 32;
    int depth = 3;
    int seed = 12345;
    int outputs = 2;
    int minBatchSize = 200;
    int labelIndex = 1;
    int trainSize = 80;
    String dataPath = "/home/bcri/train_data2/two/";
    int epochs = 10;
    int averagingFrequency = 5;
    int prefetchNumBatches = 7;



}
