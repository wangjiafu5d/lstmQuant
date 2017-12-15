package test;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import org.ujmp.core.Matrix;

import com.qq.mail271127035.BackPass;
import com.qq.mail271127035.ForwardPass;
import com.qq.mail271127035.LstmLayer;

public class QuantRunTest {
	public static void main(String[] args) {
		long start_time = System.currentTimeMillis();
		List<Matrix> xList = new ArrayList<Matrix>();
		List<Matrix> trained_list = new ArrayList<Matrix>();
		for (int i = 0; i < 6; i++) {
			xList.add(Matrix.Factory.rand(6, 1));
		}
		Matrix w_input = Matrix.Factory.randn(6, 6);
		Matrix b_input = Matrix.Factory.randn(6, 1);
		List<Matrix> w_hidden_list = new ArrayList<Matrix>();
		List<Matrix> b_hidden_list = new ArrayList<Matrix>();
		Matrix w_output = Matrix.Factory.randn(2, 6);
		Matrix b_output = Matrix.Factory.randn(2, 1);
		for (int i = 0; i < 4; i++) {
			w_hidden_list.add(Matrix.Factory.randn(1, 2));
			b_hidden_list.add(Matrix.Factory.randn(1, 6));
		}
		Matrix target = Matrix.Factory.rand(2, 1);
		System.out.println("target:" + target + "\r\n" + "++++++++++++++");
		Double learning_rate = 0.05;
		Double delta_saved = 0.0;
		for (int i = 0; i < 1000; i++) {
			ForwardPass forwardPass = new ForwardPass();
			forwardPass.add_w_input(w_input).add_b_input(b_input).add_w_hidden_list(w_hidden_list)
					.add_b_hidden_list(b_hidden_list).add_w_output(w_output).add_b_output(b_output).add_xList(xList);
			forwardPass.run();
			Matrix out = forwardPass.getOutputLayer().out;
//			 System.out.println("out:"+out+"\r\n"+"++++++++++++++");
			Matrix ht = forwardPass.getLstmLayer().ht_out_list.get(forwardPass.getLstmLayer().ht_out_list.size() - 1);
			Double sum = 1.0;
			for (int m = 0; m < ht.getRowCount(); m++) {
				for (int n = 0; n < ht.getColumnCount(); n++) {
					sum += Math.abs(ht.getAsDouble(m, n));
				}
			}
			if (sum<0.1) {
				w_input = Matrix.Factory.randn(6, 6);
				System.err.println("ht_sum : "+sum);
			}
			// System.out.println("ht: " + forwardPass.getLstmLayer().ht_out_list.get(5));
			Matrix delta_m = out.minus(target);
			Double delta = 0.5 * (Math.pow(delta_m.getAsDouble(0, 0), 2) + Math.pow(delta_m.getAsDouble(1, 0), 2));
			// System.out.println("loss"+i+" = "+delta);

			learning_rate = Math.random() / 10.0;
			System.out.println("  " + delta);
			if (delta > 5 && delta_saved - delta < 0.001) {
				learning_rate = 1.0;
			}
			delta_saved = delta;
			if (delta < 0.000000001 && i>100) {
				System.out.println("i " + i);
				System.out.println(forwardPass.getOutputLayer().out);
				for (Iterator<Matrix> iterator = trained_list.iterator(); iterator.hasNext();) {
					Matrix matrix = (Matrix) iterator.next();
					System.out.println(matrix);
				}
				break;
			}
			LstmLayer lstmLayer = forwardPass.getLstmLayer();				
			Matrix ct_out = lstmLayer.ct_out_list.get(lstmLayer.ct_out_list.size() - 1);			
			Matrix ct_prev = forwardPass.getLstmLayer().ct_out_list.get(lstmLayer.ct_out_list.size() - 2);
			Matrix ht_prev = lstmLayer.ht_out_list.get(lstmLayer.ht_out_list.size() - 2);
			
			List<Matrix> last_cell_result = lstmLayer.cells_result.get(lstmLayer.cells_result.size() - 1);
			List<Matrix> momentum = new ArrayList<Matrix>();
			momentum.add(Matrix.Factory.zeros(w_output.getRowCount(),w_output.getColumnCount()));
			momentum.add(Matrix.Factory.zeros(w_hidden_list.get(0).getRowCount(),w_hidden_list.get(0).getColumnCount()));
			momentum.add(Matrix.Factory.zeros(w_hidden_list.get(0).getRowCount(),w_hidden_list.get(0).getColumnCount()));
			momentum.add(Matrix.Factory.zeros(w_hidden_list.get(0).getRowCount(),w_hidden_list.get(0).getColumnCount()));
			momentum.add(Matrix.Factory.zeros(w_hidden_list.get(0).getRowCount(),w_hidden_list.get(0).getColumnCount()));
			momentum.add(Matrix.Factory.zeros(w_input.getRowCount(),w_input.getColumnCount()));
			BackPass backPass = new BackPass().build(ht, out, ct_out, ct_prev, ht_prev, last_cell_result, momentum, target, learning_rate ,0.0001);
			trained_list = backPass.backTrain(w_output, w_hidden_list, w_input, xList);
			// for (Iterator iterator = trained_list.iterator(); iterator.hasNext();) {
			// Matrix matrix = (Matrix) iterator.next();
			// System.out.println(matrix+"\r\n"+"++++++++++++++");
			// }
			w_output = trained_list.get(0);
			w_hidden_list.clear();
			w_hidden_list.add(trained_list.get(1));
			w_hidden_list.add(trained_list.get(2));
			w_hidden_list.add(trained_list.get(3));
			w_hidden_list.add(trained_list.get(4));
			w_input = trained_list.get(5);

		}
		long end_time = System.currentTimeMillis();
		System.out.println(end_time-start_time);
	}

}
