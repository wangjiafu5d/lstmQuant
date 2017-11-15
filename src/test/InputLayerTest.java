package test;

import java.util.ArrayList;
import java.util.List;

import org.ujmp.core.Matrix;

import com.qq.mail271127035.InputLayer;

public class InputLayerTest {
	public static void main(String[] args) {
		List<Matrix> list_x = new ArrayList<Matrix>();
		for (int i = 0; i < 6; i++) {
			list_x.add(Matrix.Factory.rand(6, 1));
		}
		Matrix w = Matrix.Factory.randn(6,6);
		Matrix b = Matrix.Factory.randn(6,1);		
		InputLayer inputLayer = InputLayer.build(w, b, list_x);
		for (int i = 0; i < inputLayer.out_list.size(); i++) {
			System.out.println(inputLayer.out_list.get(i));
		}
	}
}
