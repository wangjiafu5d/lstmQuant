package test;

import java.util.ArrayList;
import java.util.List;

public class ListTest {
	public static void main(String[] args) {
		List<StringBuilder> list = new ArrayList<StringBuilder>();
		StringBuilder sb = new StringBuilder("aabb");
		list.add(sb);
		System.out.println(list.get(0));
		sb.append("cc");
		System.out.println(list.get(0));
	}
}
