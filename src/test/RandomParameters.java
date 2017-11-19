package test;

import org.ujmp.core.Matrix;

import com.qq.mail271127035.util.FileUtil;

public class RandomParameters {
	public static void main(String[] args) {
		System.out.println(Matrix.Factory.randn(6,6));
		System.out.println(Matrix.Factory.randn(1,2));
		System.out.println(Matrix.Factory.randn(1,2));
		System.out.println(Matrix.Factory.randn(1,2));
		System.out.println(Matrix.Factory.randn(1,2));
		System.out.println(Matrix.Factory.randn(2,6));
		
		String userName = System.getenv("USERNAME");
		String desktop = "C:/Users/" + userName + "/Desktop";
		
		String s1 = desktop + "/LSTM/parameters/w_input.txt";
		String s2 = desktop + "/LSTM/parameters/b_input.txt";
		String s3 = desktop + "/LSTM/parameters/wf.txt";
		String s4 = desktop + "/LSTM/parameters/wi.txt";
		String s5 = desktop + "/LSTM/parameters/wc.txt";
		String s6 = desktop + "/LSTM/parameters/wo.txt";
		String s7 = desktop + "/LSTM/parameters/bf.txt";
		String s8 = desktop + "/LSTM/parameters/bi.txt";
		String s9 = desktop + "/LSTM/parameters/bc.txt";
		String s10 = desktop + "/LSTM/parameters/bo.txt";
		String s11 = desktop + "/LSTM/parameters/w_output.txt";
		String s12 = desktop + "/LSTM/parameters/b_output.txt";
		String s13 = desktop + "/LSTM/parameters/rate.txt";
		String s14 = desktop + "/LSTM/parameters/loss.txt";
		
		FileUtil.writeMatix(s1, Matrix.Factory.randn(6,6));
		FileUtil.writeMatix(s3, Matrix.Factory.randn(1,2));
		FileUtil.writeMatix(s4, Matrix.Factory.randn(1,2));
		FileUtil.writeMatix(s5, Matrix.Factory.randn(1,2));
		FileUtil.writeMatix(s6, Matrix.Factory.randn(1,2));
		FileUtil.writeMatix(s11, Matrix.Factory.randn(2,6));
		Matrix r = Matrix.Factory.zeros(1, 1);
		r.setAsDouble(Math.random()/1000, 0, 0);
		FileUtil.writeMatix(s13, r);
		r = Matrix.Factory.zeros(1, 1);
		r.setAsDouble(Math.random()/10, 0, 0);
		FileUtil.writeMatix(s14, r);
		
		FileUtil.writeMatix(s2, Matrix.Factory.randn(6,1));
		FileUtil.writeMatix(s7, Matrix.Factory.randn(1,6));
		FileUtil.writeMatix(s8, Matrix.Factory.randn(1,6));
		FileUtil.writeMatix(s9, Matrix.Factory.randn(1,6));
		FileUtil.writeMatix(s10, Matrix.Factory.randn(1,6));
		FileUtil.writeMatix(s12, Matrix.Factory.randn(2,1));
	}
}
