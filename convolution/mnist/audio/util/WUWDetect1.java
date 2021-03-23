package org.deeplearning4j.examples.convolution.mnist.audio.util;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.examples.convolution.mnist.audio.Maths;
import org.deeplearning4j.examples.convolution.mnist.audio.UI;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import javax.swing.*;
import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Random;

public class WUWDetect1 {


    final static int t=2;
    public static int count=0;
    public static double[][] test;
    public static double[][] mfcc=new double[127*t][13];
    static final int height = 13;
    static final int width = 254;
    static final int channels = 3;
    static byte[] b;
    public static int k;
    public static int n=0;
    static String type="wuw1";

    //2초 동안의 음성 데이터를 받는 구간

    public WUWDetect1(byte[] buffer,boolean isLastFrame) {
        //i가 초를 받는 구간
        extractMFCC(buffer,isLastFrame);
    }


    void extractMFCC(byte[] a,boolean isLastFrame) {
        try {
            test = Maths.MFCC(Maths.byteToDoubleArray(a));
            k=isLastFrame?1:0;
            for(int j=0;j<test.length;j++){
                mfcc[j+127*k]=test[j].clone();
            }
            if(isLastFrame){
                printSepctogram.saveSepctogram(mfcc,type,n);

//                System.out.println("Machine Learning....");
                if(UI.detect)
                    return;

                int p = (int) MachineLearning();

                if(p==0){//트리거워드가 맞다면

                    System.out.println("WUW1 결과 : " + p);
                    UI.detect=true;
                    System.out.println("명령어를 말하세요");
                    UI.LbNorth.setText("명령어를 말하세요");
//                   new ExtractMFCC(test,);
                }

                for (int i=0;i<253;i++){
                    for (int j=0;j<13;j++){
                        mfcc[i][j]=0;
                    }
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    double MachineLearning(){

        File file = new File(UI.BASE_PATH+type+(n)+".png");
        NativeImageLoader loader = new NativeImageLoader(height, width, channels);

        //우리는 MFCC 결과를 배열로 배꿔줘야겠지.
        INDArray testImage = null;
        try {
            testImage = loader.asMatrix(file);
        } catch (IOException e) {
            e.printStackTrace();
        }

        DataNormalization scaler = new ImagePreProcessingScaler();

        scaler.transform(testImage);

        INDArray output = UI.wuwModel.output(testImage);

//        UI.log.info("The neural nets prediction (list of probabilities per label)");
//        UI.log.info(output.toString());

        String result = output.argMax().toString();
        double totalresult = Double.parseDouble(result);
        totalresult = Math.round(totalresult);
        return totalresult;
    }
}
