import weka.core.matrix.Matrix;
//import java.io.*;
import java.util.Random;

public class assignment2 {
	
	public static Matrix data = new Matrix(new double[1000][51]);
	public static Matrix weights = new Matrix(new double[1][51]);
	public static Matrix index = new Matrix(new double[1000][1]);
	
	public static void generatedata(double a,int m,int boundary)
	{
		Random rand = new Random();
		for(int i=0;i<1000;i++)
		{
			if(i<boundary)
			{
				for(int j=0;j<m;j++)
				{
					double x = rand.nextDouble();
					double val = (x-2)*a;
					data.set(i,j,val);
				}
				for(int j=m;j<50;j++)
				{
					double x = rand.nextDouble();
					double val = (x-0.5)*4*a;
					data.set(i,j,val);
				}
				data.set(i,50,1);
			}
			else
			{
				for(int j=0;j<m;j++)
				{
					double x = rand.nextDouble();
					double val = (x+1)*a;
					data.set(i,j,val);
				}
				for(int j=m;j<50;j++)
				{
					double x = rand.nextDouble();
					double val = (x-0.5)*4*a;
					data.set(i,j,val);
				}
				data.set(i,50,1);
			}
		}
	}
	
	public static void initialize_weights(int a)
	{
		Random rand = new Random();
		for(int i=0;i<50;i++)
		{
			weights.set(0, i, (rand.nextDouble()-0.5)*a);
		}
		weights.set(0,50,1);
	}
	
	
	public static boolean check_classification(int i)
	{
		double sum=0;
		
		for(int j=0;j<51;j++)
		{
			sum+=data.get(i,j)*weights.get(0, j);
		}
		
		boolean ans=true;
		
		//System.out.println("Sum is "+sum+" for i "+i);
		
		if(sum>0)
		{
			if(i<500)
				ans = false;
			else 
				ans = true;
		}
		else
		{
			if(i<500)
				ans = true;
			else 
				ans = false;
		}
		
		return ans;
	}
	
	public static void find_misclassified()
	{
		boolean ans = true;
		for(int i=0;i<1000;i++)
		{
			index.set(i, 0, 0);
		}
		
		for(int i=0;i<1000;i++)
		{
			ans = check_classification(i);
			
			if(ans)
			{
				index.set(i, 0, 1);
			}
		}
	}
	
	
	public static void batch_perceptron_constant_rho(double rho)
	{
		initialize_weights(1000);
		int iter = 0;
		double count=1;
		while(count!=0)
		{
			iter++;
			find_misclassified();
			count=0;
			//weights.print(3, 5);
			for(int i=0;i<1000;i++)
			{
				
				if(index.get(i, 0)==1)
				{
					count++;
					if(i<500)
						weights.plusEquals(data.getMatrix(i, i, 0, 50).times(rho));
					else 
						weights.minusEquals(data.getMatrix(i, i, 0, 50).times(rho));
				}
				
			}

			//weights.print(3, 5);
			System.out.println("no. of misclassified vectors is: "+count+" iteration no. "+iter);
				
		}
		weights.print(3, 2);
	}

	public static void batch_perceptron_variable_rho()
	{
		initialize_weights(1000);
		weights.print(3,2);
		double iter = 0;
		double count=1;
		while(count!=0)
		{
			iter++;
			find_misclassified();
			count=0;
			//weights.print(3, 5);
			for(int i=0;i<1000;i++)
			{
				
				if(index.get(i, 0)==1)
				{
					count++;
					if(i<500)
						weights.plusEquals(data.getMatrix(i, i, 0, 50).times(1/iter));
					else 
						weights.minusEquals(data.getMatrix(i, i, 0, 50).times(1/iter));
				}
				
			}

			//weights.print(3, 5);
			System.out.println("no. of misclassified vectors is: "+count+" iteration no. "+iter);
				
		}
		weights.print(3, 2);
	}
		
	public static void incremental_perceptron_constant_rho(double rho)
	{
		initialize_weights(1000);
		int iter = 0;
		double count=1;
		while(count!=0)
		{
			iter++;
			count = 0;
			for(int i=0;i<1000;i++)
			{
				boolean ans = check_classification(i);
				
				if(ans)
				{
					count++;
					if(i<500)
						weights.plusEquals(data.getMatrix(i, i, 0, 50).times(rho));
					else 
						weights.minusEquals(data.getMatrix(i, i, 0, 50).times(rho));
				}
				
			}
			//weights.print(3, 5);
			System.out.println("no. of misclassified vectors is: "+count+" iteration no. "+iter);
				
		}
		weights.print(3, 2);
		
	}
	
	public static void incremental_perceptron_variable_rho()
	{
		initialize_weights(800);
		double iter = 0;
		double count=1;
		while(count!=0)
		{
			iter++;
			count = 0;
			for(int i=0;i<1000;i++)
			{
				boolean ans = check_classification(i);
				
				if(ans)
				{
					count++;
					if(i<500)
						weights.plusEquals(data.getMatrix(i, i, 0, 50).times(1/iter));
					else 
						weights.minusEquals(data.getMatrix(i, i, 0, 50).times(1/iter));
				}
				
			}
			//weights.print(3, 5);
			System.out.println("no. of misclassified vectors is: "+count+" iteration no. "+iter);
				
		}
		weights.print(3, 2);
		
	}
	
	public static void pocket_perceptron_constant_rho(int desired_unchanged_iterations,double rho)
	{
		initialize_weights(1000);

		Matrix weightscopy = new Matrix(weights.getArrayCopy());

		double misclassified = 1000;
		int no_of_unchanged_weights =0;
		
		while(no_of_unchanged_weights<desired_unchanged_iterations)
		{
			find_misclassified();
			
			double count = 0;
			
			for(int i=0;i<1000;i++)
			{
				if(index.get(i, 0)==1)
				{
					count++;
				}
			}
			
			if(count<misclassified)
			{
				no_of_unchanged_weights = 1;
				misclassified = count;
				weightscopy = new Matrix(weights.getArrayCopy());//.setMatrix(0, 0, 0, 50, weights);	
			}
			else
			{
				no_of_unchanged_weights++;
				//weights.setMatrix(0, 0, 0,50,weightscopy);
			}
			//find_misclassified();
			
			System.out.println("No of misclassified vectors "+misclassified);
	
			for(int i=0;i<1000;i++)
				{
//					if(index.get(i, 0)==1)
//					{
						if(i<500)
							weights.plusEquals(data.getMatrix(i, i, 0, 50).times(rho));
						else 
							weights.minusEquals(data.getMatrix(i, i, 0, 50).times(rho));
					//}	
				}
	
		}
		System.out.println("--------------------------------------");
		System.out.println("Converged Weights after "+no_of_unchanged_weights+ " unchanged iterations");
		weightscopy.print(3, 2);


	}

	public static void pocket_perceptron_variable_rho(int desired_unchanged_iterations)
	{
		initialize_weights(100);

		Matrix weightscopy = new Matrix(weights.getArrayCopy());

		double misclassified = 1000;
		int no_of_unchanged_weights =0;
		double iter = 0;
		
		while(!(no_of_unchanged_weights==desired_unchanged_iterations))
		{
			iter++;
			find_misclassified();
			
			double count = 0;
			
			for(int i=0;i<1000;i++)
			{
				if(index.get(i, 0)==1)
				{
					count++;
				}
			}
			
			//System.out.println("Count "+count);
			if(count<misclassified)
			{
				no_of_unchanged_weights = 0;
				misclassified = count;
				weightscopy = new Matrix(weights.getArrayCopy());

			}
			else
			{
				no_of_unchanged_weights++;
				//weights.setMatrix(0, 0, 0,50,weightscopy);
			}
			//find_misclassified();
			
			System.out.println("No of misclassified vectors "+misclassified);
			
			for(int i=0;i<1000;i++)
			{
//				if(index.get(i, 0)==1)
//				{
					if(i<500)
						weights.plusEquals(data.getMatrix(i, i, 0, 50).times(1/iter));
					else 
						weights.minusEquals(data.getMatrix(i, i, 0, 50).times(1/iter));
//				}	
			}
			
		}
		System.out.println("--------------------------------------");
		System.out.println("Converged Weights after "+no_of_unchanged_weights+ " unchanged iterations");
		weightscopy.print(3, 2);
		
	}

	public static double get_multiplication(int i)
	{
		double sum=0;
		
		for(int j=0;j<51;j++)
		{
			sum+=data.get(i,j)*weights.get(0, j);
		}
		
		if(i<500)
			return 1-sum;
		else 
			return -1-sum;
	}
	
	public static void lms_separable_rho()
	{
		initialize_weights(100);
		weights.print(3,2);
		int iter = 0;
		double count=1;
		double val;
		while(count!=0)
		{
			find_misclassified();
			
			count=0;
			for(int i=0;i<1000;i++)
			{
				if(index.get(i, 0)==1)
					count++;
			}
			
			if(count!=0)
			{
				for(int i =0;i<1000;i++)
				{	
					val = get_multiplication(i);
					//System.out.println("Value "+val);
					weights.plusEquals(data.getMatrix(i, i, 0, 50).times(0.001).times(val));
				}
			}
			
			//weights.print(3, 5);
			//System.out.println("no. of misclassified vectors is: "+count+" iteration no. "+iter);
			iter++;	
		}
		System.out.println("Converged after "+iter+" iterations");
		weights.print(3, 2);
		
	}

	public static void lms_non_separable(int desired_unchanged_iterations,double rho)
	{
		initialize_weights(100);

		Matrix weightscopy = new Matrix(weights.getArrayCopy());

		double misclassified = 1000;
		int no_of_unchanged_weights =0;
		double val;
		int iterations = 0;
		while(no_of_unchanged_weights<=desired_unchanged_iterations)
		{
			iterations++;
			//System.out.println("At start");
			//weights.print(3, 5);
			//weightscopy.print(3, 5);
			
			find_misclassified();
			
			double count = 0;
			
			for(int i=0;i<1000;i++)
			{
				if(index.get(i, 0)==1)
				{
					count++;
				}
			}
			
			if(count<misclassified)
			{
				no_of_unchanged_weights = 0;
				misclassified = count;
				//weightscopy.setMatrix(0, 0, 0, 50, weights);
				weightscopy = new Matrix(weights.getArrayCopy());
				//System.out.println("When weights are updated");
				//weights.print(3, 5);
				//weightscopy.print(3, 5);	
			}
			else
			{
				no_of_unchanged_weights++;
			}
			System.out.println("No of misclassified vectors "+misclassified +" iterations "+iterations);
			
			for(int i =0;i<1000;i++)
			{	
				//val = //get_multiplication(i);
				Matrix dp = data.getMatrix(i, i, 0, 50);
				int y ;

				if(i<500)
					y=1;
				else
					y=-1;
				val = y - dp.times(weights.transpose()).getArray()[0][0];	
				//System.out.println("Value "+val);
				weights.plusEquals(dp.times(0.001).times(val));
			}
			
			
		}
		System.out.println("--------------------------------------");
		System.out.println("Converged Weights after "+no_of_unchanged_weights+ " unchanged iterations");
		weightscopy.print(3, 2);
		
	}

	
	
	public static void main(String[] args)
	{
		//Random rand = new Random();
		double a=1;//rand.nextDouble()*1000;
		int m = 5;

		//----Separable data----
		//generatedata(a,m,500);
		
		//-----Non-Separable data----
		generatedata(a,m,540);

		//data.print(3, 5);
		double rho = 0.001;
		
		System.out.println("Start");
		
		//batch_perceptron_constant_rho(rho);
		//batch_perceptron_variable_rho();

		//incremental_perceptron_constant_rho(rho);
		//incremental_perceptron_variable_rho();
		
		//pocket_perceptron_constant_rho(400,rho);
		//pocket_perceptron_variable_rho(400);

		//lms_separable_rho();
		//lms_non_separable(400,rho);
		
		System.out.println("Done");
	}
}
