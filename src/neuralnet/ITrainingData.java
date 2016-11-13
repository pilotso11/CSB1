package neuralnet;


/**
 * Training data interface for the neural network
 */
public interface ITrainingData
{
    /**
     * Return the topology shape.
     * Must be at least 3 length.
     * {2, 3, 1} = 2 inputs, 3 hidden neurons on 1 layer, 1 output.
     * {3, 5, 5, 2} = 3 inputs, to layers of hidden neurons, with 5 neurons each, 2 outputs.
     * @return The topology shape.
     */
    int[] getTopology();

    /**
     * Are we done?
     * @return true if we are done with training
     */
    boolean isEof();

    /**
     * Return the next set of inputs.  Must be of the length of the first value in topology.
     * @return The array of inputs.
     */
    double[] getNextInputs();

    /**
     * Return the next set of target values.  Must be of the length of the last value in topology.
     * @return The array of targets.
     */
    double[] getTargetOutputs();
}

