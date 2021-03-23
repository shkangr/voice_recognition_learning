package org.deeplearning4j.examples.convolution.mnist.audio;

import java.io.IOException;
import java.util.Arrays;

import org.deeplearning4j.examples.convolution.mnist.audio.util.FFT;
import org.deeplearning4j.examples.convolution.mnist.audio.util.MFCC;

public class Maths {

	public static double[] byteToDoubleArray(byte[] b) {
		double[] d = new double[b.length];
		for (int i = 0; i < b.length; i++) {
			d[i] = b[i];
		}
		return d;
	}

	@Deprecated
	public static double[] cepstrum(double[] d) {
		double[] re = d;
		double[] im = new double[re.length];
		FFT forwardFFT = new FFT(FFT.FFT_FORWARD, 16);
		forwardFFT.transform(re, im);
		for (int i = 0; i < re.length; i++) {
			// +1 to do not have "-Infinity"
			re[i] = Math.log(Math.abs(re[i]));
		}
		forwardFFT.transform(re, im);
		return re;
	}

	public static double[] DCT(double[] d) {
		long time = System.currentTimeMillis();
		double[] re = d;
		double[] im = new double[d.length];
		FFT fourierer;
		int windowSize = re.length/50;
		fourierer = new FFT(FFT.FFT_FORWARD, re.length);
//		fourierer = new FFT(FFT.FFT_MAGNITUDE, re.length,
//				FFT.WND_HANNING, windowSize);
//		fourierer = new FFT(FFT.FFT_MAGNITUDE,16);
		fourierer.transform(re, im);
//		System.out.println(System.currentTimeMillis()-time);
		return re;
	}

	@Deprecated
	public static double[] invDCT_log_DCT(double[] d) {
		double[] re = d;
		double[] im = new double[re.length];
		FFT forwardFFT = new FFT(FFT.FFT_FORWARD, 16);
		FFT reverseFFT = new FFT(FFT.FFT_REVERSE, 16);
		forwardFFT.transform(re, im);
		for (int i = 0; i < re.length; i++) {
			re[i] = Math.log(Math.sqrt(re[i] * re[i] + im[i] * im[i]));
		}
		reverseFFT.transform(re, im);
		return re;
	}

	public static double[][] MFCC(double[] d) throws IllegalArgumentException, IOException {
		final int     NUMBERS_OF_COEFFICIENTS =    13;//32
		final int     WINDOW_SIZE             =    UI.sampleRate / 32;  //512
		final boolean USE_FIRST_COEFF         =    true;
		final double  MIN_FREQ                =    200;
		final double  MAX_FREQ                =    3500.0;
		final int     NUMBER_OF_FILTERS       =   250;
		MFCC mfcc = new MFCC(UI.sampleRate, WINDOW_SIZE,
				NUMBERS_OF_COEFFICIENTS, USE_FIRST_COEFF, MIN_FREQ, MAX_FREQ,
				NUMBER_OF_FILTERS);
		return mfcc.process(d);
	}

	@Deprecated
	public static float[] strangeMFCC(double[] d, int n) {
		float[] coeffs = new float[n];
		for (int p = 0; p < n; p++) {
			for (int k = 0; k < d.length; k++) {
				coeffs[p] += Math.log(d[k] + 1)
						* Math.cos(p * (k - 1 / 2) * Math.PI / d.length);
			}
		}
		System.out.println(Arrays.toString(coeffs)+"!!!");
		return coeffs;
	}
}
