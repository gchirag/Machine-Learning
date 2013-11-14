import weka.classifiers.Evaluation;
//import weka.classifiers.functions.LibSVM;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instances;
//import weka.core.matrix.Matrix;
import java.io.*;
import java.util.Random;
//import java.util.Random;
import weka.filters.*;
import weka.filters.unsupervised.attribute.*;;

public class Assignment3 {

	public static void main(String[] args) throws Exception {
		
		 BufferedReader reader = new BufferedReader(new FileReader("src/satimage.scale.libsvm.arff"));
    	 Instances traindata = new Instances(reader);
    	 reader.close();
    	 traindata.setClassIndex(traindata.numAttributes()-1);
    	     	 
    	 BufferedReader reader2 = new BufferedReader(new FileReader("src/satimage.scale.t.libsvm.arff"));
    	 Instances testdata = new Instances(reader2);
    	 reader2.close();
    	 testdata.setClassIndex(testdata.numAttributes()-1);
    	 
    	 BufferedReader reader3 = new BufferedReader(new FileReader("src/satimage.scale.tr.libsvm.arff"));
    	 Instances train = new Instances(reader3);
    	 reader3.close();
    	 train.setClassIndex(train.numAttributes()-1);
    	 
    	 BufferedReader reader4 = new BufferedReader(new FileReader("src/satimage.scale.val.libsvm.arff"));
    	 Instances validate = new Instances(reader4);
    	 reader4.close();
    	 validate.setClassIndex(validate.numAttributes()-1);
    	 
    	 PrintWriter writer1 = new PrintWriter("./output_cross_validation.csv");
    	 PrintWriter writer2 = new PrintWriter("./output_test.csv");
      	 //PrintWriter writer3 = new PrintWriter("./output_neuralnet.csv");
      	  
    	 
    	 for(int i=0;i<traindata.numInstances();i++)
    	 {
    		 if(traindata.instance(i).classValue()!=6)
    			 traindata.instance(i).setClassValue(1);
    		 else
    			 traindata.instance(i).setClassValue(-1);
    	 }
    	 for(int i=0;i<testdata.numInstances();i++)
    	 {
    		 if(testdata.instance(i).classValue()!=6)
    			 testdata.instance(i).setClassValue(1);
    		 else
    			 testdata.instance(i).setClassValue(-1);
    	 }
    	 
    	NumericToNominal f = new NumericToNominal();
    	f.setInputFormat(traindata);
    	f.setAttributeIndicesArray(new int[] {traindata.classIndex()});
    	traindata = Filter.useFilter(traindata,f);
    
    	NumericToNominal f2 = new NumericToNominal();
    	f2.setInputFormat(testdata);
    	f2.setAttributeIndicesArray(new int[] {testdata.classIndex()});
    	testdata = Filter.useFilter(testdata, f2);

    	NumericToNominal f3 = new NumericToNominal();
    	f3.setInputFormat(train);
    	f3.setAttributeIndicesArray(new int[] {train.classIndex()});
    	train = Filter.useFilter(train, f3);
    	
    	NumericToNominal f4 = new NumericToNominal();
    	f4.setInputFormat(validate);
    	f4.setAttributeIndicesArray(new int[] {validate.classIndex()});
    	validate = Filter.useFilter(validate, f4);
    	
    	//for(int i=11;i<100;i++)
    	//	System.out.print(i+",");
   	
    	double[] ans = new double[] {0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100};
    	double[][] ansvalue1 = new double[100][3];
    	double[][] ansvalue2 = new double[100][3];

    	for(int i=0;i<3;i++)	
    	{	
    		for(int j=0;j<100;j++)
    		{
    			System.out.println("iteration no. "+i*100+j);
	    		String[] options = new String[7];
	    		options[0] = "-Z";
	    		options[1] = "-K";
	    		options[2] = "1";
	    		options[3] = "-D";
	    		options[4] = Integer.toString(i+1);
	    		options[5] = "-C";
	    		options[6] = Double.toString(ans[j]);
	    
	    		LibSVM svm = new LibSVM();
	    		svm.setOptions(options);
	    		svm.buildClassifier(traindata);
	    		
	    		Evaluation eval1 = new Evaluation(traindata);
	    		eval1.evaluateModel(svm, traindata);
	    		ansvalue1[j][i]=eval1.pctIncorrect();
	    		
	    		Evaluation eval2 = new Evaluation(traindata);
	    		eval2.crossValidateModel(svm, testdata, 5, new Random(1));
	    		ansvalue2[j][i] = eval2.pctIncorrect();
    		}
    	}
    	
    	
    	for(int j=0;j<100;j++)
    	{
    		writer1.print(ans[j]+",");
    		writer2.print(ans[j]+",");
    	}
    	writer1.println();writer2.println();
    	
    	for(int i=0;i<3;i++)
    	{
    		for(int j=0;j<100;j++)
    		{
    			writer1.print(ansvalue1[j][i]+",");
    			writer2.print(ansvalue2[j][i]+",");
    		}
    		writer1.println();
    		writer2.println();
    	}
    	
    	writer1.close();
    	writer2.close();
    
    	/*
    	double[] ansvalue3 = new double[20];
	
    	for(int i=1;i<20;i++)
    	{
				MultilayerPerceptron neuralnet = new MultilayerPerceptron();
				String[] options = new String[4];
	    		options[0] = "-V";
	    		options[1] = "20";
	    		options[2] = "-H";
	    		options[3] = Integer.toString(i);
	    		neuralnet.setOptions(options);
	    		neuralnet.buildClassifier(traindata);
	    		
	    		Evaluation eval = new Evaluation(traindata);
	    		eval.evaluateModel(neuralnet, testdata);	
	    		ansvalue3[i] = eval.pctIncorrect();
    	}*/
 /*   	
    	for(int i=1;i<20;i++)
    	{
    		writer3.print(i+",");
    	}
    	writer3.println();
    
    	for(int i=1;i<20;i++)
    	{
    		writer3.print(ansvalue3[i]+",");
    	}
    	writer3.println();
    	writer3.close();
    	*/
	}

}
