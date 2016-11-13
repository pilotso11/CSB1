package neuralnet;

/**
 * Neural Network factory
 */
public final class NeuralNetFactory
{
    /**
     * Create a new neural network
     * Topology is an array of integrs, and must be at least size 3.
     * For example {2, 3, 1} indicates that there are 2 inputs, 1 output and 1 hidden layer with 3 nodes.
     * {3, 5, 5, 2} would be a network of 3 inputs, two hidden layers of 5 each and 2 output nodes.
     *
     * A neuron function implemntation must also be specified.
     *
     * @param topology The topology size.
     * @param neuronFunction The neuron function implementation.
     * @param biasValue The bias value (1.0 suggested).
     * @param eta The eta for the neuron weighting calculation.  Eta represents the overall net learning rate, [0.0..1.0].
     * @param alpha The alpha for the neuron weighting calculation. Alpha is the momentum impact, multiplier of last deltaWeight, [0.0..1.0].
     * @return The created network.
     */
    static public INeuralNet CreateNetwork(int[] topology, INeuronFunction neuronFunction, double biasValue, double eta, double alpha)
    {
        return new Net(topology, neuronFunction, biasValue, 100, eta, alpha);
    }
}
