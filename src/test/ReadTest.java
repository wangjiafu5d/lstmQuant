package test;

import org.ujmp.core.Matrix;
import org.ujmp.core.calculation.Calculation.Ret;

public class ReadTest {
	public static void main(String[] args) {
//		Matrix matrix = FileUtil.readMatrix("C:/Users/chuan/Desktop/data2.txt");
//		System.out.println(matrix);
		int i = 0;
		for (int j = 0; j < 6; j++) {
//			xList.add(Test.data.selectRows(Ret.LINK, i + j).transpose());
			System.out.println("i: "+ (i+j+1) + "x: "+Test.data.selectRows(Ret.LINK, i + j));
		}
		long[] column = { 1, 2 };
		Matrix target = Test.data.selectRows(Ret.LINK, 2 + 6).selectColumns(Ret.LINK, column).transpose();
		
		System.out.println("i: " + "target: "+target);
	}
}
