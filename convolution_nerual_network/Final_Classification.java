package org.deeplearning4j.examples.convolution.mnist;


import com.sun.corba.se.spi.ior.Writeable;
import javassist.Loader;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.reader.BaseRecordReader;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.ComposableRecordReader;
import org.datavec.api.records.reader.impl.csv.*;
import org.datavec.api.records.reader.impl.filebatch.FileBatchRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.api.util.ClassPathResource;

import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.INDArrayDataSetIterator;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.schedule.MapSchedule;
import org.nd4j.linalg.schedule.ScheduleType;
import org.nd4j.linalg.schedule.StepSchedule;
import org.omg.CORBA_2_3.portable.OutputStream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

import java.io.FileOutputStream;
import java.io.IOException;


public class Final_Classification {


    private static final Logger LOGGER = LoggerFactory.getLogger(Final_Classification.class);
    private static final String BASE_PATH = System.getProperty("java.io.tmpdir") + "/totalImage";


    public static void main(String[] args) throws Exception{
        int height = 13;
        int width = 254;
        int channels = 3;

        int outputNum = 5;
        int batchSize = 312;

        int nEpochs =10;

        int  seed = 1234; //test 1 = 123, 2 = 1234

        Random randNumGen = new Random(seed); // 일단 안씀

        LOGGER.info("data vertorization");

        //training data
        File trainData = new File(BASE_PATH + "/training");
        FileSplit trainSplit = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);


        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator(); // use parent directory name as the image label
        ImageRecordReader trainRR = new ImageRecordReader(height, width, channels, labelMaker);


        BalancedPathFilter pathFilter= new BalancedPathFilter(randNumGen, NativeImageLoader.ALLOWED_FORMATS, labelMaker);
        InputSplit[] filesInDirSplit = trainSplit.sample(pathFilter, 80, 20);
        InputSplit splitedtrainData = filesInDirSplit[0];
        InputSplit splitedtestData = filesInDirSplit[1];

        trainRR.initialize(splitedtrainData);
        DataSetIterator trainIter = new RecordReaderDataSetIterator(trainRR, batchSize, 1, outputNum);

        DataNormalization imageScaler = new ImagePreProcessingScaler();

        imageScaler.fit(trainIter);

        trainIter.setPreProcessor(imageScaler);
        
        

        //data for test
        File testData = new File(BASE_PATH + "/testing");
        FileSplit testSplit = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);

        ImageRecordReader testRR = new ImageRecordReader(height, width, channels, labelMaker);

        testRR.initialize(splitedtestData);
        DataSetIterator testIter = new RecordReaderDataSetIterator(testRR, batchSize, 1, outputNum);
        imageScaler.fit(testIter);
        testIter.setPreProcessor(imageScaler);

        LOGGER.info("Network configuration and training...");

        double nonZeroBias = 1;
        double dropOut = 0.5;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .weightInit(WeightInit.XAVIER)
            .activation(Activation.RELU)
            //.dist(new NormalDistribution(0.0, 0.01))
            .updater(new Nesterovs(new StepSchedule(ScheduleType.ITERATION, 1e-2,0.1,100000),0.9))
            // .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT)
            //.biasUpdater(new Nesterovs(new StepSchedule(ScheduleType.ITERATION, 2e-2, 0.1,100000), 0.9))
            .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
            .l2(5*1e-4) //L2 regularization coefficient (weights only)
            .list()
            .layer(new ConvolutionLayer.Builder(3,7)
                .nIn(channels)
                .stride(2, 2)
                .nOut(100)
                .padding(2,0)
                .build())
            .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2,2)
                .stride(1,2)
                .build()
            )
            .layer(new ConvolutionLayer.Builder(3, 5)
                .stride(2, 2) // nIn need not specified in later layers
                .nOut(250)
                .padding(2,0)
                .build())
            .layer(new LocalResponseNormalization.Builder().name("local1").build())
            .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2,2)
                .stride(2,2)
                .build()
            )
            .layer(new ConvolutionLayer.Builder(2,3)
                .padding(1,1)
                .stride(1,1)
                .nOut(400)
                .build()
            )
            .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2,2)
                .stride(1,1)
                .build()
            )
            .layer(new DenseLayer.Builder()
                .nOut(3000)
                .dropOut(dropOut)
                .build()
            )
            .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nOut(outputNum)
                .activation(Activation.SOFTMAX)
                .build())
            .setInputType(InputType.convolutional(height,width,channels))
            .build();



        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(10));

        LOGGER.info("Total num of params: {}", net.numParams());


        for(int i =0; i< 200; i++){
            net.fit(trainIter);
            Evaluation eval = net.evaluate(testIter);
            LOGGER.info("repeat num :" + i);
            LOGGER.info(eval.stats());

            trainIter.reset();
            testIter.reset();
        }

        File voiceRecogPath = new File(BASE_PATH + "/<learning_model_file>.zip");

        ModelSerializer.writeModel(net, voiceRecogPath, true);
        LOGGER.info("Model saved in -> " + voiceRecogPath.getPath() );



    }
}
