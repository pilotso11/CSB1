package neuralnet;

import java.util.Arrays;

/**
 * Neural network implementation based on David Miller's c++ at http://millermattson.com/dave.
 * See the associated video for instructions: http://vimeo.com/19569529
 */
final class Net implements INeuralNet
{
    private final int[] topology;
    private Layer[] m_layers;               // m_layers[layerNum].neurons[neuronNum]
    private double m_recentAverageError;
    private final double m_recentAverageSmoothingFactor; // Number of samples to smooth over

    // No public constructor, must be constructed from a factory
    Net(int[] topology, INeuronFunction neuronFunction, double biasValue, double averageSmoothingFactor,
            double eta, double alpha)
    {
        this.topology = topology;
        int numLayers = topology.length;
        m_layers = new Layer[numLayers];
        m_recentAverageSmoothingFactor = averageSmoothingFactor;

        // Construct the layers
        for (int layerNum = 0; layerNum < numLayers; ++layerNum)
        {
            m_layers[layerNum] = new Layer();
            int numOutputs = layerNum == topology.length - 1 ? 0 : topology[layerNum + 1];  // outputs is size of next layer

            // We have a new layer, now fill it with neurons, and add a bias neuron in each layer.
            m_layers[layerNum].neurons = new Neuron[topology[layerNum]+1];  // Add extra neuron for bias node
            for (int neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum) {
                m_layers[layerNum].neurons[neuronNum] = new Neuron(numOutputs, neuronNum, layerNum, neuronFunction, eta, alpha);
            }

            // Force the bias node's output to 1.0 (it was the last neuron pushed in this layer):
            m_layers[layerNum].neurons[topology[layerNum]-1].setOutputVal(biasValue);
        }
    }

    /**
     * Feed forward - prpogate the inputs through the network.  One pass.
     * @param inputVals The input value set.
     */
    @Override
    public double[] feedForward(final double[] inputVals)
    {
        // Check input value length
        if(inputVals.length != m_layers[0].neurons.length - 1)
            throw new IllegalArgumentException("Input length does not match expected length of " + (m_layers[0].neurons.length -1 ));

        // Assign (latch) the input values into the input neurons
        for (int i = 0; i < inputVals.length; ++i) {
            m_layers[0].neurons[i].setOutputVal(inputVals[i]);
        }

        // forward propagate
        for (int layerNum = 1; layerNum < m_layers.length; ++layerNum) {
            Layer prevLayer = m_layers[layerNum - 1];
            for (int n = 0; n < m_layers[layerNum].neurons.length - 1; ++n) {
                m_layers[layerNum].neurons[n].feedForward(prevLayer);
            }
        }
        return getResults();
    }

    /**
     * The fedback loop, reprocess the outputs and adjust the weights.
     * @param targetVals The array of targets
     */
    @Override
    public void backProp(final double[] targetVals)
    {
        // Check target value length
        if(targetVals.length != m_layers[m_layers.length-1].neurons.length - 1)
            throw new IllegalArgumentException("Target length does not match expected length of " + (m_layers[m_layers.length-1].neurons.length-1));

        // Calculate overall net error (RMS of output neuron errors)
        Layer outputLayer = m_layers[m_layers.length-1];
        double m_error = 0.0;

        for (int n = 0; n < outputLayer.neurons.length- 1; ++n) {
            double delta = targetVals[n] - outputLayer.neurons[n].getOutputVal();
            m_error += delta * delta;
        }
        m_error /= outputLayer.neurons.length - 1; // get average error squared
        m_error = Math.sqrt(m_error); // RMS

        // Implement a recent average measurement
        m_recentAverageError =
                (m_recentAverageError * m_recentAverageSmoothingFactor + m_error)
                        / (m_recentAverageSmoothingFactor + 1.0);

        // Calculate output layer gradients
        for (int n = 0; n < outputLayer.neurons.length - 1; ++n) {
            outputLayer.neurons[n].calcOutputGradients(targetVals[n]);
        }

        // Calculate hidden layer gradients
        for (int layerNum = m_layers.length - 2; layerNum > 0; --layerNum) {
            Layer hiddenLayer = m_layers[layerNum];
            Layer nextLayer = m_layers[layerNum + 1];

            for (int n = 0; n < hiddenLayer.neurons.length; ++n) {
                hiddenLayer.neurons[n].calcHiddenGradients(nextLayer);
            }
        }

        // For all layers from outputs to first hidden layer,
        // update connection weights
        for (int layerNum = m_layers.length - 1; layerNum > 0; --layerNum) {
            Layer layer = m_layers[layerNum];
            Layer prevLayer = m_layers[layerNum - 1];

            for (int n = 0; n < layer.neurons.length - 1; ++n) {
                layer.neurons[n].updateInputWeights(prevLayer);
            }
        }
    }

    /**
     * Return the outputs of a pass of the network
     * @return Array of outputs
     */
    @Override
    public double[] getResults()
    {
        // Outputs saved as output vals on the last layer
        int lastLayer = m_layers.length-1;
        int nOutputs = m_layers[lastLayer].neurons.length-1;  // Ignore bias
        double[] resultVals = new double[nOutputs];

        for (int n = 0; n < nOutputs; ++n) {
            resultVals[n] = m_layers[lastLayer].neurons[n].getOutputVal();
        }
        return resultVals;
    }

    /**
     * Return average error
     * @return Average error for reporting
     */
    @Override
    public double getRecentAverageError()
    {
        return m_recentAverageError;
    }

    @Override
    public String toString()
    {
        StringBuilder b = new StringBuilder();
        b.append("Topology: ").append(Arrays.toString(topology)).append("\nCurrent Weights:\n");
        for(int l = 0; l < m_layers.length; l++)
        {
            b.append("Layer: ").append(l).append(" --------------------------\n");
            for(Neuron n : m_layers[l].neurons)
                b.append(n.toString());
        }

        return b.toString();
    }
}

