import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.bson.BSONDecoder;
import org.bson.BSONObject;
import org.bson.BasicBSONDecoder;


public class DataReader {
    private int count;
    private int cate;
    private Map<Integer, Integer>  cates = new HashMap<>();
    private Map<Integer, ArrayList<Integer>> catesImageIds = new HashMap<>();
    public void bson(String filename) throws FileNotFoundException {
        File file = new File(filename);
        InputStream inputStream = new BufferedInputStream(new FileInputStream(file));
        BSONDecoder decoder = new BasicBSONDecoder();
        try {
            while (inputStream.available() > 0) {
                BSONObject obj = decoder.readObject(inputStream);
                if(obj == null){
                    break;
                }
                build(obj);
                count++;
            }
            PrintWriter printWriter = new PrintWriter(new File("data/count1.txt"));
            printWriter.println("number of pictures: " + count + ", number of category: " + cate );
            printWriter.flush();
            printWriter.close();
            PrintWriter printWriter2 = new PrintWriter(new File("data/cates1.txt"));
            cates.forEach((k, v) -> printWriter2.println(k + "," + v));
            printWriter2.flush();
            printWriter2.close();
            PrintWriter printWriter3 = new PrintWriter(new File("data/cates3.txt"));
            catesImageIds.forEach((k, v) -> {
                printWriter3.print(k);
                for (Integer i: v)
                    printWriter3.print("," + i);
                printWriter3.println();

            });
            printWriter3.flush();
            printWriter3.close();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            try {
                inputStream.close();
            } catch (IOException e) {
            }
        }

    }

    private void build(BSONObject record) throws IOException {
        Map<String, Object> map = record.toMap();
        //System.out.println(map.toString());
        List<Object> list = (List<Object>) map.get("imgs");
        //System.out.println(list);
        Map<String, byte[]> map1 = (Map<String, byte[]>) list.get(0);
        String image = record.get("_id").toString() + ".jpg";
        int categoryId =  Integer.parseInt(map.get("category_id").toString());
        File direct = new File("/home/bcri/train_data2/train/" + categoryId);
        if (count % 10000 == 0)
        System.err.println("number of pictures: " + count + ", number of category: " + cate );
        if (!cates.containsKey(categoryId)) {
            direct.mkdir();
            cates.put( categoryId, 0);
            catesImageIds.put(categoryId, new ArrayList<>());
            cate++;

        }
        FileOutputStream writer = new FileOutputStream(new File("/home/bcri/train_data2/train/" +categoryId+"/"+ image));
        byte[] img = map1.get("picture");
        writer.write(img);
        cates.put( categoryId, cates.get(categoryId) + 1);
        catesImageIds.get(categoryId).add(Integer.parseInt(record.get("_id").toString()));
    }
}