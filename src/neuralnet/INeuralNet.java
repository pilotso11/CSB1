package neuralnet;

/**
 * Neural Network interface
 */
public interface INeuralNet
{
    /**
     * FeedForward loop - process a set of inputs and popogate through the network to generate a set of outputs
     * @param inputVals The input value set
     * @return The output results of the run
     */
    double[] feedForward(double[] inputVals);

    /**
     * Feedback the target values to each neuron to recalibrate the network weights
     * @param targetVals The array of targets
     */
    void backProp(double[] targetVals);

    /**
     * Get outputs from most recent run
     * @return The output results
     */
    double[] getResults();

    /**
     * Get the running average error
     * @return The running average error
     */
    double getRecentAverageError();

}
