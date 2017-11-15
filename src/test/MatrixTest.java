package test;



import java.util.Arrays;

import org.ujmp.core.Matrix;
import org.ujmp.core.calculation.Calculation.Ret;
import com.qq.mail271127035.util.MathUtil;
/**
 * FileName:MatrixTest.java
 * 测试UJMP包的矩阵运算
 * 参考资料：http://blog.csdn.net/lionel_fengj/article/details/53400715
 * @author chuan
 * @Date 2017-10-29
 * @version 1.0
 */
public class MatrixTest {
	public static volatile Matrix m1;
	public static void main(String[] args) {
		int i = 2;
		int j = 3;
		int[][] a =new  int[i][j];
		for (int k = 0; k < i; k++) {
			for (int k2 = 0; k2 < j; k2++) {
				a[k][k2] = k+k2;				
			}
			
		}
		System.out.println(Arrays.deepToString(a));
		m1 = Matrix.Factory.zeros(2, 3);
		m1.setAsDouble(Math.PI, 1, 2);
		System.out.println(m1);
//		System.out.println(m1.getAsDouble(1,2)+", "+Math.PI);
		for (int k = 0; k < i; k++) {
			for (int k2 = 0; k2 < j; k2++) {
				m1.setAsDouble(a[k][k2], k, k2);				
			}
			
		}
		Matrix m2 = Matrix.Factory.ones(3, 1);// 生成全为1的3×1矩阵
		System.out.println(m1.mtimes(m2));// 矩阵相乘
		System.out.println(m1.mtimes(m2).times(10));
		System.out.println(m1.mtimes(m2).transpose());//矩阵转置
		System.out.println(m1.minus(Matrix.Factory.ones(2, 3)));// 矩阵相减
		
		System.out.println(Matrix.Factory.rand(3, 4));// 随机值为0~1的矩阵
		System.out.println(Matrix.Factory.randn(3, 4));// 随机值为-1~1的矩阵
		createMatrix(5, 6);
		Matrix m3 = Matrix.Factory.rand(1,6);
		Matrix m4 = Matrix.Factory.rand(1,6);
		Matrix m5 = Matrix.Factory.rand(1,6);
		System.out.println(MathUtil.seriesHadamard(m3,m4,m5));
		System.out.println(MathUtil.hadamard(MathUtil.hadamard(m3, m4), m5));
		
		long[] ii = {0,1};
		m5 = m5.selectRows(Ret.LINK,ii);//选取指定的行构成新矩阵
//		System.out.println(m5);
		
		Matrix matrix1 = Matrix.Factory.rand(1,2);
		Matrix matrix2 = Matrix.Factory.rand(1,2);
		Matrix matrix3 = Matrix.Factory.rand(1,2);
		matrix3 = matrix1.minus(matrix2.times(0.0));
		System.out.println(matrix1);
		System.out.println(matrix2);
		System.out.println(matrix3);
		
	}
	public static Matrix createMatrix(int m, int n) {
		return Matrix.Factory.ones(m, n);
	}
}
