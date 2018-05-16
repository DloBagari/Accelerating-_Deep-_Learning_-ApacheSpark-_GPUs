import java.io.FileNotFoundException;
public class ExtractImages {
    public static void main(String[] args) throws FileNotFoundException {
        DataReader bson = new DataReader();
        bson.bson("/home/bcri/train.bson");

    }

}
