package org.deeplearning4j.examples.convolution.mnist.audio;

import java.awt.Canvas;
import java.awt.Color;
import java.io.IOException;

import org.deeplearning4j.examples.convolution.mnist.audio.util.ExtractMFCC;

public class Graph_Painter extends Painter {

	/*private static final class Detector {
        private static int decision;
        private static int prevWeiBeg;
        private static int curWeiBeg;
        private static int slowCounter = 0;

        public static void process() {
            decision = 0x00;
            for (int i = 0; i < mfcc.length; i++) {
                curWeiBeg += mfcc[i];
            }
            curWeiBeg /= mfcc.length;

            isChanging();

            prevWeiBeg = curWeiBeg;
            voiceDecision = decision != 0x00 ? decision : voiceDecision;
        }

        private static int computeDerivation(int bound1, int bound2) {
            int res = 0;
            int b1 = bound1 < 0 ? 0 : bound1;
            int b2 = bound2 > mfcc.length ? mfcc.length : bound2;
            for (int i = b1 + 1; i < b2; i++) {
                res += mfcc[i] - mfcc[i - 1];
            }
            return res;
        }

        private static void isChanging() {
            int diff = curWeiBeg - prevWeiBeg;
            int der = computeDerivation(0, mfcc.length / 2);
            if (der < -5) {
                if (diff > 5) {
                    if (voiceDecision == FAST_MATCH) {
                        decision = SPEECH;
                        System.out.println("fast speech");
                    } else {
                        decision = FAST_MATCH;
                        System.out.println("fast");
                    }
                } else {
                    decision = SPEECH;
                }
                slowCounter = 0;
            } else
            if (diff < -5) {
                decision = SLOW_RELEASE;
                System.out.println("slow");
                if (voiceDecision == decision) {
                    slowCounter++;
                }
            } else {
                if (voiceDecision == SLOW_RELEASE || voiceDecision == SPEECH) {
                    slowCounter++;
                }
                if (slowCounter > 3) {
                    slowCounter = 0;
                    decision = NO_SPEECH;
                }
            }
        }
    }*/

    public static int voiceDecision = 0x00;
    static int drL = 1;
    static boolean useWindowFunction = true;

    private static final int SPEECH = 0x11;
    private static final int FAST_MATCH = 0x01;
    private static final int SLOW_RELEASE = 0x02;
    private static final int NO_SPEECH = 0x03;

    public static byte[] buffer;
    public static double[] fourier;
    public static double[] mfcc;
    private static byte[] a;
    public static int i=0;

    private static long timer;

    private static final int h = GraphCanvas.height;
    private static final int w = GraphCanvas.width;

    public Color Color_yellowgreen = new Color(183,240,177);
    public Color Color_yellow = new Color(250,228,0);
    public Color Color_ivory = new Color(250, 236, 197);
    public Color Color_red = new Color(241, 95, 95);
    public Color Color_green = new Color(134, 229, 127);
    public Color Color_blue = new Color(67, 116, 217, useWindowFunction ? 128 : drL == 1 ? 64 : 128);
    public Color Color_orange = new Color(255, 187, 0);
    public Color Color_pink = new Color(255, 0, 127);
    public Color Color_skyblue = new Color(0, 216, 255);

    private static final int fourierScaler = 120; //useWindowFunction ? 70 : 150;
    private static final int mfccScaler = 5;

    private static int[] approximateBuffer;
    private static int[] approximateFourier;

    private static void approxB(int x, int y) {
        approximateBuffer = new int[y];
        for (int i = 0; i < approximateBuffer.length; i++) {
            approximateBuffer[i] = (int) ((double) i / (double) y * x);
        }
    }

    private static void approxF(int x, int y) {
        approximateFourier = new int[y];
        for (int i = 0; i < approximateFourier.length; i++) {
            approximateFourier[i] = (int) ((double) i / (double) y * x);
        }
    }

    public Graph_Painter(Canvas c) {
        super(c);
    }

    @Override
    public void paint() {
        if (UI.sync) {
            synchronized (UI.locker) {
                if(!UI.s)return;
                UI.s=!UI.s;
                a = UI.buffer;
//                System.out.println("Copy");
            }
        } else {
            a = UI.buffer;
        }
        if (a == null || a.length < 1) {
            return;
        }
        g.setColor(Color_ivory);
        g.fillRect(-1, -1, w + 1, 3*h + 1);
        g.drawString(Long.toString(System.currentTimeMillis() - timer), w - 15, 20);
        drawAll(a);

        timer = System.currentTimeMillis();
    }

    void drawAll(byte[] b) {
        /*************배열의 길이 구하기************/
        // 1. Speech 배열에 값 넣기
        buffer = b;
        if (approximateBuffer == null|| approximateBuffer.length != buffer.length)
            approxB(w, buffer.length);
        int Len_buf = buffer.length - 1; // lim
        int jump = buffer.length >= w? 1 : 10;

        // 2. Fourier 배열에 값 넣기
        double[] c = Maths.DCT(Maths.byteToDoubleArray(b));
        fourier = new double[c.length / 4];
        System.arraycopy(c, 0, fourier, 0, c.length / 4);
        int Len_Fourier = fourier.length - 1;
        for(int i = 0; i<fourier.length; i++){
            fourier[i] = -Math.abs(fourier[i]);
        }
        if (approximateFourier == null || approximateFourier.length != fourier.length)
            approxF(w, fourier.length);

        // 3. Mfcc 배열에 값 넣기
        try {
            mfcc = Maths.MFCC(Maths.byteToDoubleArray(a))[0];
        } catch (IllegalArgumentException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
        int Len_Mfcc = mfcc.length / 3;

        /*************배열에  그리기************/
        double ff = 0;
        double ll = 0;

        int j = 0;
        for(int i = 0; i < Len_buf; ++i) {
            // Speech
            g.setColor(Color_green);
            g.drawLine(approximateBuffer[i], h + jump * buffer[i] + h / 2, approximateBuffer[i + drL], h + buffer[i + drL] + h / 2);

            // Fourier
            if(i<Len_Fourier){
                g.setColor(Color_red);
                g.drawLine(approximateFourier[i], (int) fourier[i] / fourierScaler + (2*h)/3 , approximateFourier[i + drL],
                    (int) fourier[i + drL] / fourierScaler +(2*h)/3);
                ff += fourier[i];
            }

            // Mfcc
            if(i<Len_Mfcc){
                g.setColor(Color.WHITE);
                g.drawLine(j, -1+ 2*h, j, h + 1 + 2*h);
                g.setColor(Color_blue);
                g.drawLine(j, (int) -mfcc[i] * mfccScaler + h / 2 + 2*h, j + 10 * drL, (int) -mfcc[i + drL] * mfccScaler + h / 2 + 2*h);
                g.drawLine(j, (int) -mfcc[i] * mfccScaler + h / 2 + 2*h + 1, j + 10 * drL, (int) -mfcc[i + drL] * mfccScaler + h / 2 + 2*h + 1);
                j += 10;
            }
        }

        ff /= fourier.length;
        g.setColor(Color.BLACK);
        int he = (int) ll + (2*h) / 3;
        g.drawLine(-1, he, w + 1, he);

        g.setColor(Color_skyblue);
        he = (int) ff / fourierScaler + (2*h) / 3;
        g.drawLine(-1, he, w + 1, he);

        // x축 그리기
        g.setColor(Color.BLACK);
        g.drawLine(-1, (3*h) / 2, w + 1, (3*h) / 2);
        g.drawLine(-1, (5*h) / 2, w + 1, (5*h) / 2);

        UI.m_PnCmd[UI.LIGHT].getGraphics().drawImage(UI.m_ArrImgcon[UI.LIGHT].getImage(), 0, 0, UI.m_PnCmd[UI.LIGHT]);
        UI.m_PnCmd[UI.CHAIR].getGraphics().drawImage(UI.m_ArrImgcon[UI.CHAIR].getImage(), 0, 0, UI.m_PnCmd[UI.CHAIR]);
        UI.m_PnCmd[UI.POWER].getGraphics().drawImage(UI.m_ArrImgcon[UI.POWER].getImage(), 0, 0, UI.m_PnCmd[UI.POWER]);
        UI.m_PnCmd[UI.SCREEN].getGraphics().drawImage(UI.m_ArrImgcon[UI.SCREEN].getImage(), 0, 0, UI.m_PnCmd[UI.SCREEN]);
        UI.m_PnCmd[UI.DOOR].getGraphics().drawImage(UI.m_ArrImgcon[UI.DOOR].getImage(), 0, 0, UI.m_PnCmd[UI.DOOR]);
    }
}
