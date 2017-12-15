package test;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import java.util.Vector;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import org.ujmp.core.Matrix;
import org.ujmp.core.calculation.Calculation.Ret;
import org.ujmp.gui.actions.SVDAction;

import com.qq.mail271127035.ForwardPass;
import com.qq.mail271127035.TrainThread;
import com.qq.mail271127035.util.FileUtil;
import com.qq.mail271127035.util.MathUtil;

public class Test {
	private static double onceLoss = 0.0;
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
	public static Matrix data = FileUtil.readMatrix(s0);
	public static Matrix w_input = FileUtil.readMatrix(s1);
	public static Matrix b_input = FileUtil.readMatrix(s2);
	public static Matrix wf = FileUtil.readMatrix(s3);
	public static Matrix wi = FileUtil.readMatrix(s4);
	public static Matrix wc = FileUtil.readMatrix(s5);
	public static Matrix wo = FileUtil.readMatrix(s6);
	public static Matrix bf = FileUtil.readMatrix(s7);
	public static Matrix bi = FileUtil.readMatrix(s8);
	public static Matrix bc = FileUtil.readMatrix(s9);
	public static Matrix bo = FileUtil.readMatrix(s10);
	public static Matrix w_output = FileUtil.readMatrix(s11);
	public static Matrix b_output = FileUtil.readMatrix(s12);

	public static List<Matrix> w_hidden_list = new ArrayList<Matrix>();
	public static List<Matrix> b_hidden_list = new ArrayList<Matrix>();
	public static List<List<Matrix>> trainedVectorLists = new Vector<List<Matrix>>();
	public static List<List<Matrix>> vt = new Vector<List<Matrix>>();
	public static List<Double> loss = new Vector<Double>();
	static {
		w_hidden_list.add(wf);
		w_hidden_list.add(wi);
		w_hidden_list.add(wc);
		w_hidden_list.add(wo);
		b_hidden_list.add(bf);
		b_hidden_list.add(bi);
		b_hidden_list.add(bc);
		b_hidden_list.add(bo);
	}

	public static void main(String[] args) {
		long start_time = System.currentTimeMillis();
//		 outAllResults(18);		
	
		long[] rows = new long[256];
		for (int i = 0; i < rows.length; i++) {
			rows[i] = i;
		}

		Matrix matrix = data.selectRows(Ret.LINK, rows);
		// predict(matrix, 240, 0.05, 12);
		// System.out.println(FileUtil.readMatrix(s13).getAsDouble(0, 0));
//		new Test().train(1, matrix, 0, 18);
		new Test().train(500000, matrix, 0.05,0.000001, 30);
//		 for (int i = 0; i < 35; i++) {
//		 System.out.println(46-i+" : "+new Test().train(4000, matrix, 0.05,0.00001, 46-i));
//		 }
//		for (int n = 0; n < 100000; n++) {
//			sgd(matrix, 128, 1000, 0.05, 18);
//		}

		long end_time = System.currentTimeMillis();
		System.out.println("程序总共用时： " + (end_time - start_time));
	}

	public static void predict(Matrix m, int stepLength, double rate,double lambda, int xListSize) {
		Test test = new Test();
		long[] rows = new long[stepLength];
		for (int step = 0; step < m.getRowCount() - stepLength + 1; step++) {
			for (int i = 0; i < stepLength; i++) {
				rows[i] = i + step;
			}
			Matrix matrix = m.selectRows(Ret.LINK, rows);
			double lossAverage = test.train(4000, matrix, rate,lambda, xListSize);

			List<Matrix> xList = new ArrayList<Matrix>();
			for (int j = 0; j < xListSize; j++) {
				xList.add(matrix.selectRows(Ret.LINK, matrix.getRowCount() - xListSize + j).transpose());
			}
			ForwardPass forwardPass = new ForwardPass();
			forwardPass.add_w_input(Test.w_input).add_b_input(Test.b_input).add_w_hidden_list(Test.w_hidden_list)
					.add_b_hidden_list(Test.b_hidden_list).add_w_output(Test.w_output).add_b_output(Test.b_output)
					.add_xList(xList);
			forwardPass.run();
			double d1 = 0;

			if (step + stepLength < m.getRowCount()) {
				d1 = m.selectRows(Ret.LINK, step + stepLength).getAsDouble(0, 0);
			}
			double d2 = forwardPass.getOutputLayer().getOut().getAsDouble(0, 0);

			double oneLoss = 0.5 * Math.pow(d1 - d2, 2);
			System.out.println(
					d2 + " " + " " + lossAverage + " " + xList.get(xListSize - 1).getAsDouble(4, 0) + " " + oneLoss);

		}

	}

	public static void sgd(Matrix data, int stepLength, int times, double rate, double lambda, int xListSize) {
		Test test = new Test();
		long[] rows = new long[stepLength];
		int start = new Random().nextInt((int) data.getRowCount() - stepLength + 1);
		for (int i = 0; i < stepLength; i++) {
			rows[i] = i + start;
		}
		Matrix matrix = data.selectRows(Ret.LINK, rows);
		System.out.println(test.train(times, matrix, rate,lambda, xListSize));

	}

	public static void outAllResults(int xListSize) {
		for (int t = 0; t < data.getRowCount() - xListSize; t++) {
			List<Matrix> xList = new ArrayList<Matrix>();
			for (int j = 0; j < xListSize; j++) {
				xList.add(data.selectRows(Ret.LINK, j + t).transpose());
			}
			ForwardPass forwardPass = new ForwardPass();
			forwardPass.add_w_input(Test.w_input).add_b_input(Test.b_input).add_w_hidden_list(Test.w_hidden_list)
					.add_b_hidden_list(Test.b_hidden_list).add_w_output(Test.w_output).add_b_output(Test.b_output)
					.add_xList(xList);
			forwardPass.run();
			System.out.println(forwardPass.getOutputLayer().getOut().transpose().toString().trim());
		}
	}

	public static void saveParameters() {
		FileUtil.writeMatix(s1, w_input);
		FileUtil.writeMatix(s3, wf);
		FileUtil.writeMatix(s4, wi);
		FileUtil.writeMatix(s5, wc);
		FileUtil.writeMatix(s6, wo);
		FileUtil.writeMatix(s11, w_output);
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

	public double train(int times, Matrix matrixData, double learning_rate, double lambda, int xListSize) {
		savedLoss = FileUtil.readMatrix(s14).getAsDouble(0, 0);
		List<Matrix> momentum = new ArrayList<Matrix>();
		momentum.add(Matrix.Factory.zeros(w_output.getRowCount(), w_output.getColumnCount()));
		momentum.add(Matrix.Factory.zeros(w_hidden_list.get(0).getRowCount(), w_hidden_list.get(0).getColumnCount()));
		momentum.add(Matrix.Factory.zeros(w_hidden_list.get(0).getRowCount(), w_hidden_list.get(0).getColumnCount()));
		momentum.add(Matrix.Factory.zeros(w_hidden_list.get(0).getRowCount(), w_hidden_list.get(0).getColumnCount()));
		momentum.add(Matrix.Factory.zeros(w_hidden_list.get(0).getRowCount(), w_hidden_list.get(0).getColumnCount()));
		momentum.add(Matrix.Factory.zeros(w_input.getRowCount(), w_input.getColumnCount()));
		for (int t = 0; t < times; t++) {
			ExecutorService exec = Executors.newFixedThreadPool(40);
			for (int i = 0; i < matrixData.getRowCount() - xListSize; i++) {
				List<Matrix> xList = new ArrayList<Matrix>();
				Matrix target = Matrix.Factory.rand(1, 1);
				for (int j = 0; j < xListSize; j++) {
					xList.add(matrixData.selectRows(Ret.LINK, i + j).transpose());
				}
				target = matrixData.selectRows(Ret.LINK, i + xListSize).selectColumns(Ret.LINK, 3);
//				long[] columns = {1,2};
//				target = matrixData.selectRows(Ret.LINK, i + xListSize).selectColumns(Ret.LINK, columns).transpose();
				TrainThread thread = new TrainThread();
				thread.setI(i);
				thread.setLearning_rate(learning_rate);
				thread.setLambda(lambda);
				thread.setxList(xList);
				thread.setTarget(target);
				thread.setMomentum(momentum);
				thread.setOut(false);
				exec.execute(thread);
			}
			exec.shutdown();
			try {
				exec.awaitTermination(2, TimeUnit.MINUTES);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}

			for (int s = 0; s < loss.size(); s++) {
				onceLoss += loss.get(s);
			}
			onceLoss = onceLoss / loss.size();
			if (onceLoss > savedLoss) {
				learning_rate = learning_rate * 0.9;				
			} else {
				if (learning_rate < 5) {
					learning_rate = learning_rate * 1.05;
				}
				
				w_output = w_output.times(0.0);
				wf = wf.times(0.0);
				wi = wi.times(0.0);
				wc = wc.times(0.0);
				wo = wo.times(0.0);
				w_input = w_input.times(0.0);
				Iterator<List<Matrix>> iterator = trainedVectorLists.iterator();
				while (iterator.hasNext()) {
					List<Matrix> trained_list = iterator.next();
					w_output = w_output.plus(trained_list.get(0));
					wf = wf.plus(trained_list.get(1));
					wi = wi.plus(trained_list.get(2));
					wc = wc.plus(trained_list.get(3));
					wo = wo.plus(trained_list.get(4));
					w_input = w_input.plus(trained_list.get(5));
				}
				Matrix v_out = Matrix.Factory.zeros(w_output.getRowCount(), w_output.getColumnCount());
				Matrix vf = Matrix.Factory.zeros(wf.getRowCount(), wf.getColumnCount());
				Matrix vi = Matrix.Factory.zeros(wi.getRowCount(), wi.getColumnCount());
				Matrix vc = Matrix.Factory.zeros(wc.getRowCount(), wc.getColumnCount());
				Matrix vo = Matrix.Factory.zeros(wo.getRowCount(), wo.getColumnCount());
				Matrix v_in = Matrix.Factory.zeros(w_input.getRowCount(), w_input.getColumnCount());
				Iterator<List<Matrix>> vt_iterator = vt.iterator();
				while (vt_iterator.hasNext()) {
					List<Matrix> vt_list = vt_iterator.next();
					v_out = v_out.plus(vt_list.get(0));
					vf = vf.plus(vt_list.get(1));
					vi = vi.plus(vt_list.get(2));
					vc = vc.plus(vt_list.get(3));
					vo = vo.plus(vt_list.get(4));
					v_in = v_in.plus(vt_list.get(5));
				}
				momentum.clear();
				double d = 1.0 / loss.size();
				w_output = w_output.times(d);
				wf = wf.times(d);
				wi = wi.times(d);
				wc = wc.times(d);
				wo = wo.times(d);
				w_input = w_input.times(d);

				momentum.add(v_out.times(d));
				momentum.add(vf.times(d));
				momentum.add(vi.times(d));
				momentum.add(vc.times(d));
				momentum.add(vo.times(d));
				momentum.add(v_in.times(d));

				// savedLoss = onceLoss;
			}
			// System.out.println("l: "+learning_rate);
			savedLoss = onceLoss;
			trainedVectorLists.clear();
			loss.clear();
			vt.clear();

			System.out.println(onceLoss + "   " + t  + "   "  +learning_rate + " " + lambda);

			if (onceLoss < 0.00001 || t % 3000 == 0) {
				saveParameters();
				saveRate(learning_rate);
				saveLoss(savedLoss);

			}
			onceLoss = 0.0;
		}
		saveParameters();
		saveRate(learning_rate);
		saveLoss(savedLoss);
		return savedLoss;
	}

}
