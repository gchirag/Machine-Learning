//import weka.classifiers.*;
import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Random;
import weka.classifiers.Evaluation;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instances;
import weka.core.Utils;
import weka.classifiers.evaluation.MarginCurve;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.SimpleCart;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
import weka.filters.unsupervised.instance.RemovePercentage;

public class Driver {
 
    public static void main(String[] args) throws Exception{
         
    	 BufferedReader reader = new BufferedReader(new FileReader("src/arrythmia.arff"));
    	 Instances data = new Instances(reader);
    	 reader.close();
    	 data.setClassIndex(data.numAttributes()-1);
    	 
    	 RemovePercentage dividedata = new RemovePercentage();
    	 dividedata.setInputFormat(data);
 		 dividedata.setOptions(Utils.splitOptions("-P 10"));
    	 Instances train = Filter.useFilter(data, dividedata);
    	 dividedata.setOptions(Utils.splitOptions("-P 10 -V"));
    	 Instances test = Filter.useFilter(data, dividedata);
    	 train.setClassIndex(train.numAttributes()-1);
    	 test.setClassIndex(test.numAttributes()-1);
    	 
    	 
    	 
    	 
    	 /*------------------- PART A,B and C----------------*/
    	 
    	/* SimpleCart tree = new SimpleCart();
    	 tree.buildClassifier(train);
    	 
    	 Evaluation eval1 = new Evaluation(train);
    	 eval1.evaluateModel(tree,test);
    	 System.out.println("Accuracy of Tree with GINI Impurity--> "+eval1.pctCorrect());
    	 
    	 
    	 eval1 = new Evaluation(train);
    	 eval1.crossValidateModel(tree, data, 5, new Random(1));
    	 System.out.println("Cross Validation Accuracy ---- Tree with GINI Impurity---> "+Math.round(eval1.pctCorrect()*100.00)/100.00);
    	 
    	 simplecartextra tree2 = new simplecartextra();
    	 tree2.buildClassifier(train);

    	 Evaluation eval2 = new Evaluation(train);
    	 eval2.evaluateModel(tree2,test);
    	 
    	 System.out.println("Accuracy of Tree using information gain--> "+eval2.pctCorrect());
    	 eval2 = new Evaluation(train);
    	 eval2.crossValidateModel(tree2, data, 5, new Random(1));
    	 System.out.println("Cross Validation Accuracy ---- Tree using information gain---> "+eval2.pctCorrect());
  
    	 RandomForest a = new RandomForest();
    	 Evaluation eval3 = new Evaluation(train);
    	 
    	 int k = 400;
    	 a = new RandomForest();
    	 a.setOptions(Utils.splitOptions("-I "+ k+ " -K 50"));
    	 a.buildClassifier(train);
    	 eval3 = new Evaluation(train);
    	 eval3.evaluateModel(a,test);
    	 
    	 double initerror = eval3.pctIncorrect(); 
    	 double posterror1=initerror, posterror2=initerror;
    	 int count = 1;
    	 if(initerror>0){
    		 while(count==1||(posterror1-posterror2)>0.01*posterror1){
    		 k=k/2;													//Binary search code
    		 a.setOptions(Utils.splitOptions("-I "+ k));
    		 a.buildClassifier(train);
        	 eval3 = new Evaluation(train);
        	 eval3.evaluateModel(a,test);
        	 posterror1 = eval3.pctIncorrect();
        	 
    		 a.setOptions(Utils.splitOptions("-I "+ k+10 ));
    		 a.buildClassifier(train);
        	 eval3 = new Evaluation(train);
        	 eval3.evaluateModel(a,test);
        	 posterror2 = eval3.pctIncorrect();

        	 count++;
        	 
        	 if((posterror1-posterror2)<0.02*posterror1){
        		 k=k+5;
        		 break;}
        	 else continue;
    		 }
    	 
    	 }
    	 
    	 //System.out.println("No. of iterations of binary search "+count);
    	 int bestk = k;
    	 System.out.println("Optimal value of k is "+k);
    	 
    	 a.setOptions(Utils.splitOptions("-I "+ bestk+ " -K 70"));
 		 
    	 a.buildClassifier(train);
    	 eval3 = new Evaluation(train);
    	 eval3.evaluateModel(a,test);
         System.out.println("Accuracy of Random forest------>  "+eval3.pctCorrect());
    	 
    	 a.buildClassifier(train);
    	 eval3 = new Evaluation(train);
    	 eval3.crossValidateModel(a, data, 5, new Random(1));
    	 //System.out.println(eval3.toSummaryString());
    	 System.out.println("Cross Validation Accuracy ---- Random Forest---> "+eval3.pctCorrect());

    	 
    	 partd(train,test);																/*----------PART D---------*/
  //  	 parte(data,bestk);									/*----------PART E---------*/
		 RandomForest a = new RandomForest();
    	 a.setNumTrees(30);
    	 a.setNumFeatures(70);
    	 a.buildClassifier(train);
    	 System.out.println("Out Of Bag Error---> "+a.measureOutOfBagError());			/*----------PART F---------*/
    	 
    	 
    	 a.setNumTrees(30);
    	 a.setNumFeatures(70);
    	 a.buildClassifier(train);
    	 
    	 Evaluation eval3 = new Evaluation(train);
    	 eval3.evaluateModel(a, test);
    	 
    	 FastVector p = eval3.predictions();
    	 
    	 partg(p,test.numInstances());													/*-------------PART G-------------*/
    	 
    	 
    	 
}
    
public static void partg(FastVector p, int n){
	MarginCurve m = new MarginCurve();
	Instances margindata = m.getCurve(p);
	double s = 0, s_square = 0 ;
	for (int i=0;i<margindata.numInstances();i++){
		double marginval = margindata.instance(i).value(0);
		int numberofpoints = (int)margindata.instance(i).value(0);
		
		s = s + marginval * (double)numberofpoints;
		s_square = s_square + marginval * marginval * (double)numberofpoints;
	}
	double strength = s/(double)n;
	double s_square_expected  = s_square/(double)n;
	double var = s_square_expected - strength*strength;
	double correlation = var/(1.0-strength*strength);
	double ans = var/(strength*strength);
	
	System.out.println("Upper bound on random forest classifier error is "+ans);
	System.out.println("Strength is "+strength);
	System.out.println("Correlation is "+correlation);
}
public static void parte(Instances data, int bestk){
try {

		RemovePercentage dividedata = new RemovePercentage();
		dividedata.setInputFormat(data);
		dividedata.setOptions(Utils.splitOptions("-P 10"));
   	 	Instances train = Filter.useFilter(data, dividedata);
   	 	dividedata.setOptions(Utils.splitOptions("-P 10 -V"));
   	 	Instances test = Filter.useFilter(data, dividedata);
   	 	
   	 	Instances testcopy = test;

   	 	double[] numericarray;
		double[] correctness = new double[test.numAttributes()];
   	 	
   	 	for(int i=0;i<test.numAttributes()-1;i++){
   	 		test=testcopy;
	   	 	
	   	 	Attribute a = train.attribute(i);

	   	 	if(test.attribute(i).isNominal()){
	            @SuppressWarnings("unchecked")
				ArrayList<String> listofvalues = Collections.list(a.enumerateValues());
	            
	            for(int j=0; j<test.numInstances(); j++){
	                String  replacement  =  (String) listofvalues.get((int)(Math.random() * listofvalues.size()));
	                test.instance(j).setValue(i, replacement);
	            }
	        }
	        else if(test.attribute(i).isNumeric()){
	        	double replacement;
	        	numericarray = train.attributeToDoubleArray(i);
				
	        	for(int j=0; j<test.numInstances(); j++){
	    	        	Random random = new Random();
	    	        	replacement = numericarray[random.nextInt(numericarray.length)];
	    	        	test.instance(j).setValue(i, replacement);		
	            }
	            
	        }
	   	 
	   	 	RandomForest forest = new RandomForest();
    	 
	   	 	
	   	 	forest.setOptions(Utils.splitOptions("-I "+ bestk + " -K 70"));
	   	 	forest.buildClassifier(train);
	   	 	Evaluation eval = new Evaluation(train);

	   	 	eval.evaluateModel(forest,test);
	   	 	//System.out.println("Random Forest---> attribute no. "+i+" percentage correct "+eval.pctCorrect());
	   	 	
	   	 	correctness[i] = eval.pctCorrect(); 
	   	 	
   	 	}
   	 	
   	 	int maxindex=0,maxindex2=0,maxindex3=0;
   	 	
   	 	for(int i=1;i<test.numAttributes()-1;i++){
   	 		if(correctness[i]<correctness[maxindex]){
   	 			maxindex3=maxindex2;
   	 			maxindex2=maxindex;
   	 			maxindex=i;
   	 		}
   	 		else if(correctness[i]>=correctness[maxindex]&&correctness[i]<correctness[maxindex2]){
   	 			maxindex3 = maxindex2;
   	 			maxindex2 = i;
   	 		}
   	 		else if(correctness[i]>=correctness[maxindex2]&&correctness[i]<correctness[maxindex3]){
   	 			maxindex3 = i;
   	 		}
   	 		else continue;
   	 	}
   	 	
   	 	System.out.println("Best Attribute is attribute no. "+ maxindex +" Percentage correctness is "+ correctness[maxindex]);
   	 	System.out.println("Second Best Attribute is attribute no. "+ maxindex2 +" Percentage correctness is "+ correctness[maxindex2]);
   	 	System.out.println("Third Best Attribute is attribute no. "+ maxindex3 +" Percentage correctness is "+ correctness[maxindex3]);
   	 	
   	 	
   	 	
} catch (Exception e) {
		e.printStackTrace();
	}
}
    
    public static void partd( Instances train, Instances test)
	{
    	Instances traincopy = train;
    	Instances testcopy = test;
    	
    	try{
    		
		double[] median 		= new double[train.numAttributes()];
		double[] numericarray 	= new double[train.numAttributes()];
		
		for(int i=0;i<train.numAttributes();i++)
		{
			if(train.attribute(i).isNominal())
				median[i]=train.meanOrMode(i);
			else if(train.attribute(i).isNumeric()){
				numericarray = train.attributeToDoubleArray(i);
				Arrays.sort(numericarray);
				median[i]=Utils.kthSmallestValue(numericarray, numericarray.length/2);
			}
		}		
		
		//int count = 0;
		
		for(int i=0;i<train.numInstances();i++){
    		 for(int j=0;j<train.numAttributes();j++){
    			 if(train.instance(i).isMissing(j)){
    				 //count ++;
    				 train.instance(i).setValue(j, median[j]);
    			 }
    		 }
    	 }
		
		for(int i=0;i<test.numInstances();i++){
   		 for(int j=0;j<test.numAttributes();j++){
   			 if(test.instance(i).isMissing(j)){
   				 //count ++;
   				 test.instance(i).setValue(j, median[j]);
   				 }
   		 }
   	 	}
    	
		
		SimpleCart tree = new SimpleCart();
   	 	tree.buildClassifier(train);
		
   	 	Evaluation eval1 = new Evaluation(train);
   	 	eval1.evaluateModel(tree,test);
   	    
   	 	System.out.println("Accuracy of Tree after replacing missing values with median--> "+eval1.pctCorrect());
   	 	
		
   	 	ReplaceMissingValues replace = new ReplaceMissingValues();
   	 	replace.setInputFormat(traincopy);
   	 	SimpleCart tree2 = new SimpleCart();
   	 	FilteredClassifier fc = new FilteredClassifier();
   	 	fc.setFilter(replace);
   	 	fc.setClassifier(tree2);
   	 	fc.buildClassifier(traincopy);
   	 	int count=0;
   	 	for(int i=0; i<testcopy.numInstances();i++){
   	 		double pred = fc.classifyInstance(testcopy.instance(i));
   	 		if(testcopy.classAttribute().value((int)testcopy.instance(i).classValue())==testcopy.classAttribute().value((int)pred))
   	 				count++;
   	 	}
   	 	
   	 	System.out.println("Accuracy of Tree after replacing missing values with mean--> "+100.0*count/testcopy.numInstances());
   	  	
    	} 
   	 	catch (Exception e) {
			e.printStackTrace();
		}   	 
	}

}
    