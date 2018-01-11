package test;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import org.ujmp.core.Matrix;
import org.ujmp.core.calculation.Calculation.Ret;

import com.qq.mail271127035.GruTrainThread;
import com.qq.mail271127035.util.FileUtil;
import com.qq.mail271127035.util.MyMatrixUtil;

public class TestGru {
	private static List<List<Matrix>> gradList = new ArrayList<List<Matrix>>();
	private static double loss = 0.0;
	private static double savedLoss = 0.0;
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
	private static Matrix data = FileUtil.readMatrix(s0);
	private static Matrix w_input = FileUtil.readMatrix(s1);
	private static Matrix b_input = FileUtil.readMatrix(s2);
	private static Matrix wf = FileUtil.readMatrix(s3);
	private static Matrix wi = FileUtil.readMatrix(s4);
	private static Matrix wc = FileUtil.readMatrix(s5);
	private static Matrix bf = FileUtil.readMatrix(s7);
	private static Matrix bi = FileUtil.readMatrix(s8);
	private static Matrix bc = FileUtil.readMatrix(s9);
	private static Matrix w_output = FileUtil.readMatrix(s11);
	private static Matrix b_output = FileUtil.readMatrix(s12);

	private static List<Matrix> w_hidden_list = new ArrayList<Matrix>();
	private static List<Matrix> b_hidden_list = new ArrayList<Matrix>();
	private static List<List<Double>> loss1 = new ArrayList<List<Double>>();
	private static List<List<Double>> loss2 = new ArrayList<List<Double>>();
	static {
		w_hidden_list.add(wf);
		w_hidden_list.add(wi);
		w_hidden_list.add(wc);
		b_hidden_list.add(bf);
		b_hidden_list.add(bi);
		b_hidden_list.add(bc);
	}

	public static void main(String[] args) {
		double start = System.currentTimeMillis();
		long[] rows = new long[256];
		for (int i = 0; i < rows.length; i++) {
			rows[i] = i;
		}

		Matrix matrix = data.selectRows(Ret.LINK, rows);
		train(100000, matrix, 0.005, 0.000001, 42);
		double end = System.currentTimeMillis();
		System.out.println(end - start);
	}

	public static double train(int times, Matrix matrixData, double learning_rate, double lambda, int xListSize) {

		savedLoss = FileUtil.readMatrix(s14).getAsDouble(0, 0);
		Matrix vWInput = MyMatrixUtil.copyZerosMatrix(w_input);
		Matrix vBInput = MyMatrixUtil.copyZerosMatrix(b_input);
		Matrix vWz = MyMatrixUtil.copyZerosMatrix(wf);
		Matrix vWr = MyMatrixUtil.copyZerosMatrix(wi);
		Matrix vWo = MyMatrixUtil.copyZerosMatrix(wc);
		Matrix vBz = MyMatrixUtil.copyZerosMatrix(bf);
		Matrix vBr = MyMatrixUtil.copyZerosMatrix(bi);
		Matrix vBo = MyMatrixUtil.copyZerosMatrix(bc);
		double r = 0.9;
		Matrix vWOutput = MyMatrixUtil.copyZerosMatrix(w_output);
		Matrix vBOutput = MyMatrixUtil.copyZerosMatrix(b_output);
		for (int t = 0; t < times; t++) {

			ExecutorService exec = Executors.newFixedThreadPool(9);
			for (int i = 0; i < matrixData.getRowCount() - xListSize; i++) {
				List<Matrix> xList = new ArrayList<Matrix>();
				List<Matrix> targetList = new ArrayList<Matrix>();
				List<Matrix> gradSumList = new ArrayList<Matrix>();
				List<Double> oncePassLoss1 = new ArrayList<Double>();
				List<Double> oncePassLoss2 = new ArrayList<Double>();
				gradList.add(gradSumList);
				loss1.add(oncePassLoss1);
				loss2.add(oncePassLoss2);
				for (int j = 0; j < xListSize; j++) {
					xList.add(matrixData.selectRows(Ret.LINK, i + j).transpose());
				}

				targetList.add(matrixData.selectRows(Ret.LINK, i + xListSize - 2).selectColumns(Ret.LINK, 3));
				targetList.add(matrixData.selectRows(Ret.LINK, i + xListSize - 1).selectColumns(Ret.LINK, 3));
				targetList.add(matrixData.selectRows(Ret.LINK, i + xListSize).selectColumns(Ret.LINK, 3));
//				 long[] columns = { 1, 2 };
//				 targetList.add(matrixData.selectRows(Ret.LINK, i +
//				 xListSize).selectColumns(Ret.LINK, columns).transpose());
				GruTrainThread thread = new GruTrainThread();
				thread.build(w_input, b_input, w_hidden_list, b_hidden_list, w_output, b_output, xList, 3, targetList,
						gradSumList, oncePassLoss1, oncePassLoss2);
				exec.execute(thread);
			}
			exec.shutdown();
			try {
				exec.awaitTermination(2, TimeUnit.SECONDS);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}

			Matrix d1 = MyMatrixUtil.copyZerosMatrix(gradList.get(0).get(0));
			Matrix d2 = MyMatrixUtil.copyZerosMatrix(gradList.get(0).get(1));
			Matrix d3 = MyMatrixUtil.copyZerosMatrix(gradList.get(0).get(2));
			Matrix d4 = MyMatrixUtil.copyZerosMatrix(gradList.get(0).get(3));
			Matrix d5 = MyMatrixUtil.copyZerosMatrix(gradList.get(0).get(4));
			Matrix d6 = MyMatrixUtil.copyZerosMatrix(gradList.get(0).get(5));
			Matrix d7 = MyMatrixUtil.copyZerosMatrix(gradList.get(0).get(6));
			Matrix d8 = MyMatrixUtil.copyZerosMatrix(gradList.get(0).get(7));
			Matrix d9 = MyMatrixUtil.copyZerosMatrix(gradList.get(0).get(8));
			Matrix d10 = MyMatrixUtil.copyZerosMatrix(gradList.get(0).get(9));
			for (int i = 0; i < gradList.size(); i++) {
				List<Matrix> list = gradList.get(i);
				d1 = d1.plus(list.get(0));
				d2 = d2.plus(list.get(1));
				d3 = d3.plus(list.get(2));
				d4 = d4.plus(list.get(3));
				d5 = d5.plus(list.get(4));
				d6 = d6.plus(list.get(5));
				d7 = d7.plus(list.get(6));
				d8 = d8.plus(list.get(7));
				d9 = d9.plus(list.get(8));
				d10 = d10.plus(list.get(9));
			}

			double l1 = 0.0;
			double l2 = 0.0;
			for (int i = 0; i < loss1.size(); i++) {
				l1 += loss1.get(i).get(0);
				l2 += loss2.get(i).get(0);
			}
			loss = l1 / loss1.size();
			double onceLoss2 = l2 / loss2.size();
			if (loss < savedLoss) {
				learning_rate = learning_rate * 1.03;
			} else {
				learning_rate = learning_rate * 0.9;
			}
//  m = m*r + (grad+w * w.norm2*λ )*η*(1-r)
			double element = learning_rate * (1 - r);
			vWInput = vWInput.times(r)
					.plus(d9.plus(w_input.times(w_input.norm2() * lambda)).times(element));
			vBInput = vBInput.times(r)
					.plus(d10.plus(b_input.times(b_input.norm2() * lambda)).times(element));
			vWz = vWz.times(r).plus(d3.plus(wf.times(wf.norm2() * lambda)).times(element));
			vWr = vWr.times(r).plus(d4.plus(wi.times(wi.norm2() * lambda)).times(element));
			vWo = vWo.times(r).plus(d5.plus(wc.times(wc.norm2() * lambda)).times(element));
			vBz = vBz.times(r).plus(d6.plus(bf.times(bf.norm2() * lambda)).times(element));
			vBr = vBr.times(r).plus(d7.plus(bi.times(bi.norm2() * lambda)).times(element));
			vBo = vBo.times(r).plus(d8.plus(bc.times(bc.norm2() * lambda)).times(element));
			vWOutput = vWOutput.times(r)
					.plus(d1.plus(w_output.times(w_output.norm2() * lambda)).times(element));
			vBOutput = vBOutput.times(r)
					.plus(d2.plus(b_output.times(b_output.norm2() * lambda)).times(element));
			w_input = w_input.minus(vWInput);
			b_input = b_input.minus(vBInput);
			wf = wf.minus(vWz);
			wi = wi.minus(vWr);
			wc = wc.minus(vWo);
			bf = bf.minus(vBz);
			bi = bi.minus(vBr);
			bc = bc.minus(vBo);
			w_output = w_output.minus(vWOutput);
			b_output = b_output.minus(vBOutput);

			savedLoss = loss;

			loss1.clear();
			loss2.clear();
			gradList.clear();
			w_hidden_list.clear();
			w_hidden_list.add(wf);
			w_hidden_list.add(wi);
			w_hidden_list.add(wc);
			b_hidden_list.clear();
			b_hidden_list.add(bf);
			b_hidden_list.add(bi);
			b_hidden_list.add(bc);
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
		FileUtil.writeMatix(s7, bf);
		FileUtil.writeMatix(s8, bi);
		FileUtil.writeMatix(s9, bc);
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
