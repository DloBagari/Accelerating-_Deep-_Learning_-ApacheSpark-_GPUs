
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.LearningRatePolicy;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.HashMap;
import java.util.Map;

public class CnnConfiguration {
    private final int depth;
    private final int width;
    private final int height;
    private final int numOutput;
    private final int seed;
    public CnnConfiguration(int depth, int width1, int height1, int numOutput1, int seed1){
        this.depth = depth;
        this.width = width1;
        this.height = height1;
        this.numOutput = numOutput1;
        this.seed = seed1;
    }

    public MultiLayerConfiguration getConf() {
        Map<Integer, Double> lrSchedule = new HashMap<>();
        lrSchedule.put(0, 0.003);
        //drop rate in the end of training, if training takes 6000 iteration
        lrSchedule.put(40, 0.001);
        lrSchedule.put(80, 0.0005);
        lrSchedule.put(100, 0.0001);
        return new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(1)
                .regularization(true).l2(0.0005)
                .learningRate(0.003)
                //.biasLearningRate(0.02)
                .learningRateDecayPolicy(LearningRatePolicy.Schedule)
                .learningRateSchedule(lrSchedule)
                //.lrPolicyPower(0.75)
                .weightInit(WeightInit.RELU)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.ADAM)
                .list()
                .layer(0, new ConvolutionLayer.Builder(3,3)
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
                        .nIn(depth)
                        .stride(1, 1)
                        .nOut(64)
                        .activation(Activation.LEAKYRELU)
                        .build())
                .layer(1, new ConvolutionLayer.Builder(3,3)
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
                        .stride(1, 1)
                        .nOut(64)
                        .activation(Activation.LEAKYRELU)
                        .build())
                .layer(2, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(3, new ConvolutionLayer.Builder(3,3)
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
                        .stride(1, 1)
                        .nOut(128)
                        .activation(Activation.LEAKYRELU)
                        .build())
                .layer(4, new ConvolutionLayer.Builder(3,3)
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
                        .stride(1, 1)
                        .nOut(128)
                        .activation(Activation.LEAKYRELU)
                        .build())
                .layer(5, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(6, new ConvolutionLayer.Builder(3,3)
                        .stride(1, 1)
                        .nOut(240)
                        .activation(Activation.LEAKYRELU)
                        .build())
                .layer(7, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())

                .layer(8 , new DenseLayer.Builder()
                        .activation(Activation.LEAKYRELU)
                        .nOut(1000)
                        .build())
                .layer(9 , new DenseLayer.Builder()
                        .activation(Activation.LEAKYRELU)
                        .nOut(1000)
                        .build())

                //softMax needs LossFunction.MCXENT
                .layer( 10, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .nOut(numOutput)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutional(height, width, depth))
                .backprop(true)
                .pretrain(false)
                .build();
    }

}