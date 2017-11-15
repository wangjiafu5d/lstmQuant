package test;

import java.util.ArrayList;
import java.util.List;

import org.ujmp.core.Matrix;

import com.qq.mail271127035.LstmCell;

public class LstmCellTest {
	public static void main(String[] args) {
		Matrix x1 = Matrix.Factory.randn(1, 6);
		Matrix h_in = Matrix.Factory.zeros(1, 6);
		Matrix ct_in = Matrix.Factory.zeros(1, 6);
		List<Matrix> wList = new ArrayList<Matrix>();
		List<Matrix> bList = new ArrayList<Matrix>();
		for (int i = 0; i < 4; i++) {
			wList.add(Matrix.Factory.randn(1,2));
			bList.add(Matrix.Factory.randn(1,6));
		}
		LstmCell lstmCell = LstmCell.build(wList, bList);
		List<Matrix> list = new ArrayList<Matrix>();
		list = lstmCell.lstmCell_Out(x1, h_in, ct_in);
		System.out.println("ct_out "+list.get(0)+"\r\n"+"ht "+list.get(1)+"\r\n");
	}
}
