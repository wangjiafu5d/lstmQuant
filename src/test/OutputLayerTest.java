package test;

import org.ujmp.core.Matrix;

import com.qq.mail271127035.OutputLayer;

public class OutputLayerTest {
	public static void main(String[] args) {
		Matrix x = Matrix.Factory.rand(1,6);
		x = x.transpose();
		Matrix w = Matrix.Factory.randn(2,6);
		Matrix b = Matrix.Factory.rand(2,1);
		OutputLayer outputLayer = OutputLayer.build(w, b, x);
		System.out.println(w);
		System.out.println(b);
		System.out.println(x);
		System.out.println(outputLayer.out);
	}
}
