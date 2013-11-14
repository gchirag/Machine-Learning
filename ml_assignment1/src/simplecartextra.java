import weka.classifiers.trees.SimpleCart;

public class simplecartextra extends SimpleCart{

	private static final long serialVersionUID = 1L;

public double computeGini(double[] dist, double total){
	if (total==0) return 0;
    double val = 0;
    for (int i=0; i<dist.length; i++) {
      val += (dist[i]/total)*Math.log(dist[i]/total)/Math.log(2.0);
    }
    return -val;
}
}