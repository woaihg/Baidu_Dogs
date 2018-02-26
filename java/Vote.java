import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.io.IOException;
import java.io.FileReader;
import java.util.*;  

//public class Func{
public class Vote{
	public static void main(String[] args){
		//Map<String,Integer> aMap = null;
		//aMap = getTrainList("trainList.txt");
		//readTxtByLine(aMap,"train.txt",args[0]);
		//changeRes(args[0]);
		Map<String,Map<String,Integer>> aMap = new HashMap<>();
		readBatch(aMap,args[0]);
		printRes(aMap,args[0]);
	}
	public static void printRes(Map<String,Map<String,Integer>> aMap,String batchFilePath){
		try{
			File file = new File(batchFilePath);
			BufferedReader reader = new BufferedReader(new FileReader(file));
			String fileName = null;
			while((fileName = reader.readLine()) != null){
				printItems(aMap,fileName);
				break;
			}
		}catch(IOException e){

		}finally{

		}
	}
	public static void readBatch(Map<String,Map<String,Integer>> aMap,String batchFilePath){
		try{
			File file = new File(batchFilePath);
			BufferedReader reader = new BufferedReader(new FileReader(file));
			String fileName = null;
			while((fileName = reader.readLine()) != null){
				//System.out.println(fileName);
				countItems(aMap,fileName);
			}
		}catch(IOException e){

		}finally{

		}
	}
	public static void printItems(Map<String,Map<String,Integer>> aMap,String batchFilePath){
		try{
			File file = new File(batchFilePath);
			BufferedReader reader = new BufferedReader(new FileReader(file));
			String tempString = null;
			while((tempString = reader.readLine()) != null){
				String[] tempInfo = tempString.split(",");
				String idLeft = tempInfo[0];//.substring(1,idLeft.length());
				idLeft = tempInfo[0].substring(1,idLeft.length());
				String idRight = tempInfo[1];//.substring(0,idRight.length()-1);
				idRight = tempInfo[1].substring(0,idRight.length()-1);
				String label = tempInfo[2];
				
				String id = idLeft+","+idRight;
				String voteLabel = aMap.get(id).get("max_id")+"";
				System.out.println(voteLabel+"\t"+id);
			}
			reader.close();
		}catch(IOException e){
			e.printStackTrace();
		}finally{

		}
	}
	public static void countItems(Map<String,Map<String,Integer>> aMap,String filePath){
		try{
			File file = new File(filePath);
			BufferedReader reader = new BufferedReader(new FileReader(file));
			String tempString = null;
			while((tempString = reader.readLine()) != null){
				String[] tempInfo = tempString.split(",");
				String idLeft = tempInfo[0];//.substring(1,idLeft.length());
				idLeft = tempInfo[0].substring(1,idLeft.length());
				String idRight = tempInfo[1];//.substring(0,idRight.length()-1);
				idRight = tempInfo[1].substring(0,idRight.length()-1);
				String label = tempInfo[2];
				String id = idLeft+","+idRight;
				if(!aMap.containsKey(id)){
					aMap.put(id,new HashMap<>());
					aMap.get(id).put(label,1);
					aMap.get(id).put("max_id",Integer.parseInt(label));
					aMap.get(id).put("max_val",1);
				}else{
					if(!aMap.get(id).containsKey(label)){
						aMap.get(id).put(label,1);
					}else{
						int val = aMap.get(id).get(label) + 1;
						aMap.get(id).put(label,val);
						if(val > aMap.get(id).get("max_val")){
							aMap.get(id).put("max_id",Integer.parseInt(label));
							aMap.get(id).put("max_val",val);
						}

					}
				}
				//System.out.println(label+"\t"+idLeft.substring(1,idLeft.length())+","+idRight.substring(0,idRight.length()-1));
			}
			reader.close();
		}catch(IOException e){
			e.printStackTrace();
		}finally{

		}
	}
	public static void changeRes(String filePath){
		try{
			File file = new File(filePath);
			BufferedReader reader = new BufferedReader(new FileReader(file));
			String tempString = null;
			while((tempString = reader.readLine()) != null){
				String[] tempInfo = tempString.split(",");
				String idLeft = tempInfo[0];
				String idRight = tempInfo[1];
				String label = tempInfo[2];
				System.out.println(label+"\t"+idLeft.substring(1,idLeft.length())+","+idRight.substring(0,idRight.length()-1));
			}
			reader.close();
		}catch(IOException e){
			e.printStackTrace();
		}finally{

		}
	}
	public static void readTxtByLine(Map<String,Integer> aMap,String filePath,String cho){
		try{
			File file = new File(filePath);
			BufferedReader reader = new BufferedReader(new FileReader(file));
			String tempString = null;
			while((tempString = reader.readLine()) != null){
				String[] tempInfo = tempString.split(" ");
				String id = tempInfo[0];
				String label = tempInfo[1];
				if(aMap.containsKey(id+".jpg")){
					if(cho.equals("id")){
						System.out.println(id);
					}else{
						System.out.println(label);
					}
				}	

			}
			reader.close();
		}catch(IOException e){
			e.printStackTrace();
		}finally{

		}
	}
	public static Map getTrainList(String filePath){
		Map<String,Integer> aMap = new HashMap<String,Integer>();
		try{
			File file = new File(filePath);
			BufferedReader reader = new BufferedReader(new FileReader(file));
			String tempString = null;
			while((tempString = reader.readLine()) != null){
				aMap.put(tempString,-1);
			}
		}catch(IOException e){
			e.printStackTrace();
		}finally{

		}
		return aMap;
	}
}