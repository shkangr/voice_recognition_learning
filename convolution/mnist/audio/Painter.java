package org.deeplearning4j.examples.convolution.mnist.audio;


abstract class Painter implements Runnable {

	protected static java.awt.Graphics g;
	private static java.awt.image.BufferStrategy bufferStrategy;
	protected long frames;

	static java.awt.Canvas canvas;

	private static boolean precompute() {
		try {
			bufferStrategy = canvas.getBufferStrategy();
			if (bufferStrategy == null) {
				canvas.createBufferStrategy(2);
				canvas.requestFocus();
				return true;
			}
			g = bufferStrategy.getDrawGraphics();
			return false;
		} catch (Exception e) {
			return true;
		}
	}

	protected static void show() {
		g.dispose();
		bufferStrategy.show();
	}
	public Painter(java.awt.Canvas c) {
		canvas = c;
	}

	public abstract void paint();

	public boolean render() {
		if (precompute()) {
			// some magic hiding here
			return false;
		}
		paint();
		show();
		return true;
	}

	@Override
	public void run() {
		while (true) {
			startPainting();
		}
	}

	private void startPainting() {
		if (render()) {
			show();
		} else {
			startPainting();
		}
	}

}
