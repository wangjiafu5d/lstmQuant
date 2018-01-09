package com.qq.mail271127035.util;

import org.ujmp.core.Matrix;

public class MyMatrixUtil {
	public static Matrix copyZerosMatrix(Matrix m) {
		return Matrix.Factory.zeros(m.getRowCount(), m.getColumnCount());
	};

	public static Matrix copyOnesMatrix(Matrix m) {
		return Matrix.Factory.ones(m.getRowCount(), m.getColumnCount());
	}

}
