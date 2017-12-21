package com.qq.mail271127035;

import org.ujmp.core.Matrix;

public class OutputLayer {
	//2×1矩阵
	public Matrix out;

	public static OutputLayer build(final Matrix w, final Matrix b, final Matrix x) {
		OutputLayer outputLayer = new OutputLayer();
		Matrix result = InputCell.build(w, b).out(x).transpose();
		outputLayer.out = result;
		return outputLayer;
	}

	public Matrix getOut() {
		return out;
	}

	public void setOut(Matrix out) {
		this.out = out;
	}
}
