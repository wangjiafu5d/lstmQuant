package test;

import java.util.ArrayList;
import java.util.List;

import org.ujmp.core.Matrix;

import com.qq.mail271127035.LstmLayer;

public class LstmLayerTest {
	public static void main(String[] args) {
		List<Matrix> xList = new ArrayList<Matrix>();
		List<Matrix> wList = new ArrayList<Matrix>();
		List<Matrix> bList = new ArrayList<Matrix>();
		for (int i = 0; i < 6; i++) {
			xList.add(Matrix.Factory.randn(1,6));
		}
		for (int i = 0; i < 4; i++) {
			wList.add(Matrix.Factory.randn(1,2));
			bList.add(Matrix.Factory.randn(1,6));
		}
		LstmLayer lstmLayer = LstmLayer.build(wList, bList, xList);
		for (int i = 0; i < lstmLayer.ct_out_list.size(); i++) {
			System.out.println(lstmLayer.ct_out_list.get(i));
			System.out.println(lstmLayer.ht_out_list.get(i));
		}
		System.out.println(lstmLayer.cells_result.size());
		System.out.println(lstmLayer.cells_result.get(0).size());
		System.out.println(lstmLayer.cells_result.get(0).get(5));
		System.out.println(lstmLayer.cells_result.get(1).get(5));
		System.out.println(lstmLayer.cells_result.get(2).get(5));
	}
}
