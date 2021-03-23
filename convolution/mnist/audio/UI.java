package org.deeplearning4j.examples.convolution.mnist.audio;

import org.deeplearning4j.examples.convolution.mnist.audio.util.ExtractMFCC;
import org.deeplearning4j.examples.convolution.mnist.audio.util.WUWDetect1;
import org.deeplearning4j.examples.convolution.mnist.audio.util.WUWDetect2;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.awt.BorderLayout;
import java.awt.Canvas;
import java.awt.Color;
import java.awt.Container;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.GridLayout;
import java.awt.Image;
import java.awt.Label;
import java.awt.Window;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;
import javax.lang.model.type.ExecutableType;
import javax.sound.sampled.*;
import javax.swing.ImageIcon;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.border.LineBorder;

public class UI extends JFrame {
    private static final long serialVersionUID = 1L;

    public static boolean running = true;

    public static int height = 256;
    public static int width = 640;
    public static int Btnstart_w = 60;
    public static int Btnstart_h = 32;

    public static String SPACE ="&nbsp;";
    public static String TAP ="&nbsp; &nbsp; ";

    public Font Font_Italic = new Font("Serif", Font.ITALIC, 16);
    public Font Font_BigItalic = new Font("Serif", Font.ITALIC, 32);

    public Color Color_yellowgreen = new Color(183,240,177);
    public Color Color_yellow = new Color(250,244,192);

    public LineBorder Border_line = new LineBorder(Color_yellowgreen,1);

    private GraphCanvas m_graphcanvas = null;
    private Container m_content = null;

    public static final int LIGHT = 0;
    public static final int CHAIR = 1;
    public static final int POWER = 2;
    public static final int SCREEN = 3;
    public static final int DOOR = 4;

    public final static JPanel[] m_PnCmd ={ new JPanel(), new JPanel(), new JPanel(), new JPanel(), new JPanel()};
    public static Graphics m_arrG[] = new Graphics[5];
    public static ImageIcon[] m_ArrImgcon = new ImageIcon[5];
    public static String m_beforeImgPath[] = new String[5];
    public static String m_afterImgPath[] = new String[5];

    public static JButton Btnstart;
    public static JLabel LbNorth;
    public static JPanel PnNorth;

    static Canvas canvas = new Canvas() {
        private static final long serialVersionUID = 1L;
        {setPreferredSize(new Dimension(width, height));}
    };

    public UI(){
        super("speech recognition");
        setDefaultCloseOperation(EXIT_ON_CLOSE);
        addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent e) {
                super.windowClosing(e);
                System.exit(0);
            }
        });
        setLocation(0, 200);
        setLayout(new BorderLayout());

        if(m_content == null)
            m_content = getContentPane();
        m_content.setBackground(Color_yellow);

        m_content.add(canvas, BorderLayout.CENTER);
        setResizable(true);

        ///////// 시작화면(왼쪽) //////////
        PnNorth = new JPanel();
        PnNorth.setBackground(Color_yellow);
        PnNorth.setPreferredSize(new Dimension(width, Btnstart_h * 6));

        LbNorth = new JLabel();
        LbNorth.setText("<html><br>"+TAP+"Speech<br> Recognition<html>");
        LbNorth.setAlignmentX(Label.CENTER);
        LbNorth.setAlignmentY(Label.CENTER);
        LbNorth.setFont(Font_BigItalic);
        PnNorth.add(LbNorth);

        final JPanel PnSouth = new JPanel();
        PnSouth.setBackground(Color_yellow);
        PnSouth.setPreferredSize(new Dimension(width, Btnstart_h * 3/2));

        Btnstart = new JButton("Start");
        Btnstart.setPreferredSize(new Dimension(Btnstart_w, Btnstart_h));
        Btnstart.setFont(Font_Italic);
        Btnstart.setBackground(Color_yellowgreen);
        Btnstart.setBorder(Border_line);
        Btnstart.setEnabled(true);


        ///////// 그래프 화면(오른쪽) //////////
        prepareResources();

        final JPanel PnImage = new JPanel();
        PnImage.setBackground(Color_yellow);
        PnImage.setLayout(new GridLayout(1, 5));

        m_PnCmd[UI.LIGHT].setPreferredSize(new Dimension(0, 0));
        m_PnCmd[UI.CHAIR].setPreferredSize(new Dimension(0, 0));
        m_PnCmd[UI.POWER].setPreferredSize(new Dimension(0, 0));
        m_PnCmd[UI.SCREEN].setPreferredSize(new Dimension(0, 0));
        m_PnCmd[UI.DOOR].setPreferredSize(new Dimension(0, 0));

        // 1. Light 2. Chair 3. Power 4. Screen 5. Door
        m_PnCmd[LIGHT].setBackground(Color_yellow);
        m_PnCmd[CHAIR].setBackground(Color_yellow);
        m_PnCmd[POWER].setBackground(Color_yellow);
        m_PnCmd[SCREEN].setBackground(Color_yellow);
        m_PnCmd[DOOR].setBackground(Color_yellow);

        PnImage.add(m_PnCmd[LIGHT]);
        PnImage.add(m_PnCmd[CHAIR]);
        PnImage.add(m_PnCmd[POWER]);
        PnImage.add(m_PnCmd[SCREEN]);
        PnImage.add(m_PnCmd[DOOR]);

        // 배치 시작
//        PnSouth.add(Btnstart, BorderLayout.CENTER);
        m_content.add(PnNorth, BorderLayout.NORTH);
        m_content.add(PnSouth, BorderLayout.SOUTH);
        m_content.add(PnImage, BorderLayout.CENTER);

        m_graphcanvas = new GraphCanvas();
        Dimension Dms_Img = new Dimension(128, 128);
        m_PnCmd[LIGHT].setPreferredSize(Dms_Img);
        m_PnCmd[CHAIR].setPreferredSize(Dms_Img);
        m_PnCmd[POWER].setPreferredSize(Dms_Img);
        m_PnCmd[SCREEN].setPreferredSize(Dms_Img);
        m_PnCmd[DOOR].setPreferredSize(Dms_Img);

        // 이벤트 리스너 생성
        ActionListener startListener = new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                if (Btnstart.getText().equals("Start")) {
                    //버튼 누르고 몇 초 후에 음성 입력을 받을 건지

                    Btnstart.setText("Stop");
                    LbNorth.setText("Speech Recognition");
                    PnNorth.setPreferredSize(new Dimension(width, Btnstart_h * 2));
                    if(m_graphcanvas== null){
                        m_graphcanvas = new GraphCanvas();
                    }
                    Dimension Dms_Img = new Dimension(128, 128);
                    m_PnCmd[UI.LIGHT].setPreferredSize(Dms_Img);
                    m_PnCmd[UI.CHAIR].setPreferredSize(Dms_Img);
                    m_PnCmd[UI.POWER].setPreferredSize(Dms_Img);
                    m_PnCmd[UI.SCREEN].setPreferredSize(Dms_Img);
                    m_PnCmd[UI.DOOR].setPreferredSize(Dms_Img);
                    pack();
                } else if(Btnstart.getText().equals("Stop")) {
                    Btnstart.setText("Start");
                    //LbNorth.setText("<html><br>"+TAP+"Speech<br> Recognition<html>");
                    pack();
                }
//                ExtractMFCC.noMike=!ExtractMFCC.noMike;
            }
        };
        // 이벤트 리스너 등록
//        Btnstart.addActionListener(startListener);
        pack();
        setVisible(true);
    }
    public static void stopUI(){
        Btnstart.setEnabled(false);
//        LbNorth.setText("Speech Recognition");
        // PnNorth.setPreferredSize(new Dimension(width, Btnstart_h * 6));

    }
    public static void resumeUI(){
        Btnstart.setEnabled(true);
        Btnstart.setText("Start");
        LbNorth.setText("Speech Recognition");
        // PnNorth.setPreferredSize(new Dimension(width, Btnstart_h * 6));
    }

    private void prepareResources() {
        // m_beforeImgPath
        m_beforeImgPath[UI.LIGHT] = System.getProperty("user.dir")+"\\pictures\\lightOff.png";
        m_beforeImgPath[UI.CHAIR] = System.getProperty("user.dir")+"\\pictures\\chairUP.png";
        m_beforeImgPath[UI.POWER] = System.getProperty("user.dir")+"\\pictures\\powerOn.png";
        m_beforeImgPath[UI.SCREEN] = System.getProperty("user.dir")+"\\pictures\\screenOff.png";
        m_beforeImgPath[UI.DOOR] = System.getProperty("user.dir")+"\\pictures\\doorClose.png";

        // m_afterImgPath
        m_afterImgPath[UI.LIGHT] = System.getProperty("user.dir")+"\\pictures\\lightOn.png";
        m_afterImgPath[UI.CHAIR] = System.getProperty("user.dir")+"\\pictures\\chairDown.png";
        m_afterImgPath[UI.POWER] = System.getProperty("user.dir")+"\\pictures\\powerOff.png";
        m_afterImgPath[UI.SCREEN] = System.getProperty("user.dir")+"\\pictures\\screenLoad.png";
        m_afterImgPath[UI.DOOR] = System.getProperty("user.dir")+"\\pictures\\doorOpen.png";

        loadImgIcon();
    }
    private void loadImgIcon() {
        m_ArrImgcon[LIGHT] = new ImageIcon(m_beforeImgPath[LIGHT]);
        m_ArrImgcon[CHAIR] = new ImageIcon(m_beforeImgPath[CHAIR]);
        m_ArrImgcon[POWER] = new ImageIcon(m_beforeImgPath[POWER]);
        m_ArrImgcon[SCREEN] = new ImageIcon(m_beforeImgPath[SCREEN]);
        m_ArrImgcon[DOOR] = new ImageIcon(m_beforeImgPath[DOOR]);
    }

    /**//* Audio format settings */
    /**/public static int sampleRate = //32
        //64
        //128
        //256
        //512
        //1024
        //2048
        //4096
        //8192
        //11025
        16384//���Ӱ��� ������ ����ϴ� ��
//			16000
        //22050 //IS NOT POWER OF 2
        //32768
        //44100 //IS NOT POWER OF 2
        //65536
        ;
    /**/public static int sampleSizeInBits = 16;
    /**/public static int channels = 1;  //������ ���� �� �ȵǸ� 2�� �����
    /**/public static boolean signed = true;
    /**/public static boolean bigEndian = false;//true; //������ ���� ��
    /**//* ======================= */

    public static final byte BUFFER = 1;

    //    public static boolean running;
    public static boolean end=true;
    public static int state;
    public static boolean sync = true;
    public static boolean order = true;
    public static byte buffer[];
    public static Object locker = new Object();

    public static byte audio[];
    public static ByteArrayOutputStream rec;
    static  TargetDataLine line;
    static AudioFormat format;

    public static final String BASE_PATH = System.getProperty("user.dir")+"/";
    static File locationToSave ;
    static File locationToSavewuw ;
    public static MultiLayerNetwork model;
    public static MultiLayerNetwork wuwModel;

    public static boolean isLastFrame=false;
    public static boolean isLastForWUW1=false;
    public static boolean isLastForWUW2=true;
    public static boolean detect=false;
    public static boolean s=true;
    public static int wait=0;

    public static void main(String args[]) {

        /*if(args.length!=2){
            System.err.println("명령어 Model, Trigger Word Model");
            return;
        }*/
        ModelInit();
        captureAudio();
        new UI();


    }


    private static void ModelInit(){
        locationToSave= new File(BASE_PATH+"CommandRecognitionModel.zip");
        locationToSavewuw=new File(BASE_PATH+"TriggerWordModel.zip");

        System.out.println(BASE_PATH);

        if (locationToSave.exists()&&locationToSavewuw.exists()) {
            System.err.println("Saved Model Found!");
        } else {
            System.err.println("File not found!");
            System.err.println("This example depends on running MyClassifier, run that example first");
            System.exit(0);
        }
        try {
            model= ModelSerializer.restoreMultiLayerNetwork(locationToSave);
            wuwModel=ModelSerializer.restoreMultiLayerNetwork(locationToSavewuw);
        } catch (IOException e) {
            e.printStackTrace();
        }

    }
    private static void captureAudio() {
        try {
            format = getFormat();

            DataLine.Info info = new DataLine.Info(TargetDataLine.class, format);
            line = ((TargetDataLine) AudioSystem.getLine(info));
            line.open(format);
            line.start();
            new Thread(new Runnable() {
                int bufferSize = (int) format.getSampleRate()* format.getFrameSize();// 16384 * 2
                @Override
                public void run() {
                    buffer = new byte[bufferSize];

                                while(end) {
                                    while (running) {
                                        synchronized (locker) {
                                            if(s)continue;
                                            s=!s;

                                if(detect&&!isLastFrame){
                                    //여기 명령어 말하라고해

                                    try {
                                        Thread.sleep(1000);
                                    } catch (InterruptedException e) {
                                        e.printStackTrace();
                                    }
                                }
                                line.read(buffer, 0, buffer.length);
                                if(!detect) {
                                    //여기 명령어 말하라고 한거 없애
                                    LbNorth.setText("");
                                    wait=0;
                                    new Thread(new Runnable() {
                                        @Override
                                        public void run() {
                                            new WUWDetect1(buffer, isLastForWUW1 = !isLastForWUW1);
                                        }
                                    }).start();
                                    new Thread(new Runnable() {
                                        @Override
                                        public void run() {
                                            new WUWDetect2(buffer, isLastForWUW2 = !isLastForWUW2);
                                        }
                                    }).start();
                                }else{
                                    if(wait++==0)continue;
                                    new Thread(new Runnable() {
                                        @Override
                                        public void run() {
                                            new ExtractMFCC(buffer, isLastFrame = !isLastFrame);
                                        }
                                    }).start();
                                    clearBuffer();
                                }
//                                System.out.println("Capture");
                            }
                        }
                    }
                }
            }).start();

        } catch (LineUnavailableException e) {
            System.err.println("Line unavailable: " + e);
            System.exit(-2);
        }
    }

    private static AudioFormat getFormat() {
        return new AudioFormat(sampleRate, sampleSizeInBits, channels, signed, bigEndian);
    }
    private static void clearBuffer(){
        for (int i=0;i<253;i++){
            for (int j=0;j<13;j++){
                WUWDetect1.mfcc[i][j]=0;
                WUWDetect2.mfcc[i][j]=0;
            }
        }
    }
}
