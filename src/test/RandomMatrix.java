package test;

import org.ujmp.core.Matrix;

public class RandomMatrix {
	public static void main(String[] args) {
		System.out.println(Matrix.Factory.randn(6,6));
		System.out.println(Matrix.Factory.randn(1,2));
		System.out.println(Matrix.Factory.randn(2,6));
	}
}
