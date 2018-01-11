package test;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.List;

public class ListTest {
	public static void main(String[] args) {
		List<StringBuilder> list = new ArrayList<StringBuilder>();
		StringBuilder sb = new StringBuilder("aabb");
		list.add(sb);
		System.out.println(list.get(0));
		sb.append("cc");
		sb = new StringBuilder("0.56747");
		sb.append("cc");
		System.out.println(list.get(0));
		String s = "0.555";
		List<String> listD = new ArrayList<String>();
		listD.add(s);
		s.substring(0, 3);
		System.out.println(listD.get(0));
		
		
	}
}
