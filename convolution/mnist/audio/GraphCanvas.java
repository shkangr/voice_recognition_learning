package org.deeplearning4j.examples.convolution.mnist.audio;

import java.awt.BorderLayout;
import java.awt.Canvas;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Frame;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;

public class GraphCanvas extends Frame {
	private static final long serialVersionUID = 1L;
	
	public static int height = 256;
	public static int width = 1024;
	
	static Canvas canvas = new Canvas() {
		private static final long serialVersionUID = 1L;
		{setPreferredSize(new Dimension(width, height * 3));	}
	};
	
	public GraphCanvas() {
		super("Canvas"); // Fourier Transform 
		addWindowListener(new WindowAdapter() {
			@Override
			public void windowClosing(WindowEvent e) {
				super.windowClosing(e);
				System.exit(0);
			}
		});
		setLocation(UI.width + 20, 0);
		setLayout(new BorderLayout());
		add(canvas, BorderLayout.CENTER);
		setResizable(true);
		(new Thread(new Graph_Painter(canvas))).start();
		pack();
		setVisible(true);
	}
}
