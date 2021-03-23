package org.deeplearning4j.examples.convolution.mnist.audio.util;
import org.deeplearning4j.examples.convolution.mnist.audio.UI;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileWriter;

import javax.imageio.ImageIO;



public class printSepctogram {

    public static Color getColor(double power) {
        double H,S,B;
        S=1;
        B=1;
        if(10<Math.abs(power)){
            H=Math.abs(power)*20;
        }else{
            H = Math.abs(power) *36;
            S=((power*0.1)+5)/10;

        }// Hue (note 0.4 = Green, see huge chart below)
        if(power<0){
            B=0.5;
        }
        if(power==0){
            B=1;
            S=0;
        }

        return Color.getHSBColor((float)H/360, (float)S, (float)B);
    }
    public static void saveSepctogram(double mfcc[][],String type,int n){
        try{
            int nX = 127*ExtractMFCC.t;
            int nY = 13;
            double[][] plotData = new double[nX][nY];
            StringBuilder sb=new StringBuilder();

            BufferedImage theImage = new BufferedImage(nX,nY,BufferedImage.TYPE_INT_RGB);
            double ratio;
            for(int x = 0; x<nX; x++){
                for(int y = 0; y<nY; y++){
                    ratio = mfcc[x][y];
//                    sb.append(String.format("%.4f",mfcc[x][y])+" ");
                    sb.append(ratio+" ");

                    Color newColor = getColor(ratio);
                    theImage.setRGB(x, y, newColor.getRGB());
                }
                sb.append("\r\n");
            }
//            System.out.println();
            File outputfile = new File(UI.BASE_PATH+type+n+".png");

//            FileWriter fw=new FileWriter(UI.BASE_PATH+type+n+".txt");
//            fw.write(sb.toString());
            ImageIO.write(theImage, "png", outputfile);

//            fw.close();

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}

