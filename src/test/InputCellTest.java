package test;

import org.ujmp.core.Matrix;

import com.qq.mail271127035.InputCell;
/**
 * 
 * 输入层神经元矩阵运算测试
 * @author chuan
 * @Date 2017-10-29
 * @version 1.0
 *
 */
public class InputCellTest {
	public static void main(String[] args) {
		Matrix w = Matrix.Factory.randn(5, 5);
		Matrix x = Matrix.Factory.rand(5, 1);
		
//		x.setAsDouble(1, 0, 0);
//		x.setAsDouble(0.5, 1, 0);
//		x.setAsDouble(1, 2, 0);
//		x.setAsDouble(1, 3, 0);
//		x.setAsDouble(0.5, 4, 0);
//		x.setAsDouble(1, 5, 0);
		Matrix b = Matrix.Factory.rand(5, 1);
		System.out.println(x);
		System.out.println(w);
		System.out.println(b);
		InputCell inputCell = InputCell.build(w, b);
		Matrix out = inputCell.inputCell_Out(x);
		
//		String s = out.toString();
//		String[] str = s.split(" ");
//		for (int i = 0; i < str.length; i++) {
//			String string = str[i];
//			System.out.print(string+", ");
//		}
		System.out.println(out);
		
	}
}
