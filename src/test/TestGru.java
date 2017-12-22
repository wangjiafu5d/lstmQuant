package test;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Vector;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import org.ujmp.core.Matrix;
import org.ujmp.core.calculation.Calculation.Ret;

import com.qq.mail271127035.GruTrainThread;
import com.qq.mail271127035.util.FileUtil;

public class TestGru {
	public static List<List<Matrix>> gradList = new Vector<List<Matrix>>();
	private static double loss = 0.0;
	private static double savedLoss = 0.0;
	private static double tryLoss = 0.0;
	public static String s0;
	public static String s1;
	public static String s2;
	public static String s3;
	public static String s4;
	public static String s5;
	public static String s6;
	public static String s7;
	public static String s8;
	public static String s9;
	public static String s10;
	public static String s11;
	public static String s12;
	public static String s13;
	public static String s14;
	static {
		initFileAddress();
	}
	public static Matrix data = FileUtil.readMatrix(s0);
	public static Matrix w_input = FileUtil.readMatrix(s1);
	public static Matrix b_input = FileUtil.readMatrix(s2);
	public static Matrix wf = FileUtil.readMatrix(s3);
	public static Matrix wi = FileUtil.readMatrix(s4);
	public static Matrix wc = FileUtil.readMatrix(s5);
	public static Matrix w_output = FileUtil.readMatrix(s11);
	public static Matrix b_output = FileUtil.readMatrix(s12);

	public static List<Matrix> w_hidden_list = new ArrayList<Matrix>();
	public static List<Double> loss1 = new Vector<Double>();
	public static List<Double> loss2 = new Vector<Double>();
	static {
		w_hidden_list.add(wf);
		w_hidden_list.add(wi);
		w_hidden_list.add(wc);
	}

	public static void main(String[] args) {
		double start = System.currentTimeMillis();
		long[] rows = new long[256];
		for (int i = 0; i < rows.length; i++) {
			rows[i] = i;
		}

		Matrix matrix = data.selectRows(Ret.LINK, rows);
		train(10000, matrix, 0.001, 0.0, 18);
		double end = System.currentTimeMillis();
		System.out.println(end - start);
	}

	public static double train(int times, Matrix matrixData, double learning_rate, double lambda, int xListSize) {
		savedLoss = FileUtil.readMatrix(s14).getAsDouble(0, 0);
		Matrix vWInput = Matrix.Factory.zeros(w_input.getRowCount(), w_input.getColumnCount());
		Matrix vBInput = Matrix.Factory.zeros(b_input.getRowCount(), b_input.getColumnCount());
		Matrix vWz = Matrix.Factory.zeros(wf.getRowCount(), wf.getColumnCount());
		Matrix vWr = Matrix.Factory.zeros(wi.getRowCount(), wi.getColumnCount());
		Matrix vWo = Matrix.Factory.zeros(wc.getRowCount(), wc.getColumnCount());
		double r = 0.9;
		Matrix vWOutput = Matrix.Factory.zeros(w_output.getRowCount(), w_output.getColumnCount());
		Matrix vBOutput = Matrix.Factory.zeros(b_output.getRowCount(), b_output.getColumnCount());
		for (int t = 0; t < times; t++) {
			ExecutorService exec = Executors.newFixedThreadPool(40);
			for (int i = 0; i < matrixData.getRowCount() - xListSize; i++) {
				List<Matrix> xList = new ArrayList<Matrix>();
				List<Matrix> targetList = new ArrayList<Matrix>();
				for (int j = 0; j < xListSize; j++) {
					xList.add(matrixData.selectRows(Ret.LINK, i + j).transpose());
				}

				// targetList.add(matrixData.selectRows(Ret.LINK, i + xListSize -
				// 2).selectColumns(Ret.LINK, 3));
				// targetList.add(matrixData.selectRows(Ret.LINK, i + xListSize -
				// 1).selectColumns(Ret.LINK, 3));
				targetList.add(matrixData.selectRows(Ret.LINK, i + xListSize).selectColumns(Ret.LINK, 3));
				// long[] columns = {1,2};
				// target = matrixData.selectRows(Ret.LINK, i +
				// xListSize).selectColumns(Ret.LINK, columns).transpose();
				GruTrainThread thread = new GruTrainThread();
				thread.build(w_input, b_input, w_hidden_list, w_output, b_output, xList, 1, targetList);
				exec.execute(thread);
			}
			exec.shutdown();
			try {
				exec.awaitTermination(2, TimeUnit.MINUTES);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
			Matrix d1 = gradList.get(0).get(0).times(0);
			Matrix d2 = gradList.get(0).get(1).times(0);
			Matrix d3 = gradList.get(0).get(2).times(0);
			Matrix d4 = gradList.get(0).get(3).times(0);
			Matrix d5 = gradList.get(0).get(4).times(0);
			Matrix d6 = gradList.get(0).get(5).times(0);
			Matrix d7 = gradList.get(0).get(6).times(0);
			for (int i = 0; i < gradList.size(); i++) {
				List<Matrix> list = gradList.get(i);
				d1 = d1.plus(list.get(0));
				d2 = d2.plus(list.get(1));
				d3 = d3.plus(list.get(2));
				d4 = d4.plus(list.get(3));
				d5 = d5.plus(list.get(4));
				d6 = d6.plus(list.get(5));
				d7 = d7.plus(list.get(6));
			}

			double l1 = 0.0;
			double l2 = 0.0;
			for (int i = 0; i < loss1.size(); i++) {
				l1 += loss1.get(i);
				l2 += loss2.get(i);
			}
			loss = l1 / loss1.size();
			double onceLoss2 = l2 / loss2.size();

			if (Math.abs(1 - loss / savedLoss) > 0.02) {
				learning_rate = learning_rate * 0.5;
			} else {
				if (Math.abs(1 - loss / savedLoss) < 0.005) {
					learning_rate = learning_rate * 1.01;
				}
				vWInput = vWInput.times(r).plus(d6.plus(w_input.times(w_input.norm2() * lambda)).times(learning_rate));
				vBInput = vBInput.times(r).plus(d7.plus(b_input.times(b_input.norm2() * lambda)).times(learning_rate));
				vWz = vWz.times(r).plus(d3.plus(wf.times(wf.norm2() * lambda)).times(learning_rate));
				vWr = vWr.times(r).plus(d4.plus(wi.times(wi.norm2() * lambda)).times(learning_rate));
				vWo = vWo.times(r).plus(d5.plus(wc.times(wc.norm2() * lambda)).times(learning_rate));
				vWOutput = vWOutput.times(r)
						.plus(d1.plus(w_output.times(w_output.norm2() * lambda)).times(learning_rate));
				vBOutput = vBOutput.times(r)
						.plus(d2.plus(b_output.times(b_output.norm2() * lambda)).times(learning_rate));
				w_input = w_input.minus(vWInput);
				// System.out.println(d1);
				// System.out.println(d1.norm2());
				// b_input = b_input.minus(vBInput);
				wf = wf.minus(vWz);
				wi = wi.minus(vWr);
				wc = wc.minus(vWo);
				w_output = w_output.minus(vWOutput);
				// b_output = b_output.minus(vBOutput);

			}
			savedLoss = loss;

			loss1.clear();
			loss2.clear();
			gradList.clear();
			w_hidden_list.clear();
			w_hidden_list.add(wf);
			w_hidden_list.add(wi);
			w_hidden_list.add(wc);
			System.out.println(loss + "   " + t + "   " + learning_rate + " " + lambda + " " + onceLoss2);

			if (loss < 0.00001 || t % 3000 == 0) {
				saveParameters();
				saveRate(learning_rate);
				saveLoss(savedLoss);

			}

		}
		saveParameters();
		saveRate(learning_rate);
		saveLoss(savedLoss);
		return savedLoss;
	}

	public static void saveParameters() {
		FileUtil.writeMatix(s1, w_input);
		FileUtil.writeMatix(s2, b_input);
		FileUtil.writeMatix(s3, wf);
		FileUtil.writeMatix(s4, wi);
		FileUtil.writeMatix(s5, wc);
		FileUtil.writeMatix(s11, w_output);
		FileUtil.writeMatix(s12, b_output);
	}

	public static void saveRate(double rate) {
		Matrix r = Matrix.Factory.zeros(1, 1);
		r.setAsDouble(rate, 0, 0);
		FileUtil.writeMatix(s13, r);
	}

	public static void saveLoss(double savedloss) {
		Matrix r = Matrix.Factory.zeros(1, 1);
		r.setAsDouble(savedloss, 0, 0);
		FileUtil.writeMatix(s14, r);
	}

	public static void initFileAddress() {
		String userName = System.getenv("USERNAME");
		String desktop = "C:/Users/" + userName + "/Desktop";
		s0 = desktop + "/data5.txt";
		s1 = desktop + "/LSTM/parameters/w_input.txt";
		s2 = desktop + "/LSTM/parameters/b_input.txt";
		s3 = desktop + "/LSTM/parameters/wf.txt";
		s4 = desktop + "/LSTM/parameters/wi.txt";
		s5 = desktop + "/LSTM/parameters/wc.txt";
		s6 = desktop + "/LSTM/parameters/wo.txt";
		s7 = desktop + "/LSTM/parameters/bf.txt";
		s8 = desktop + "/LSTM/parameters/bi.txt";
		s9 = desktop + "/LSTM/parameters/bc.txt";
		s10 = desktop + "/LSTM/parameters/bo.txt";
		s11 = desktop + "/LSTM/parameters/w_output.txt";
		s12 = desktop + "/LSTM/parameters/b_output.txt";
		s13 = desktop + "/LSTM/parameters/rate.txt";
		s14 = desktop + "/LSTM/parameters/loss.txt";
	}

}
