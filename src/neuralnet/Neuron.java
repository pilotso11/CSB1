package neuralnet;

/**
 * Neuron Class
 */
final class Neuron
{
    private final double eta;    // overall net learning rate, [0.0..1.0]
    private final double alpha;   // momentum, multiplier of last deltaWeight, [0.0..1.0]

    private double m_outputVal;
    private int m_myLayer;
    private int m_myIndex;
    private double m_gradient;

    private Connection[] m_outputWeights;

    private INeuronFunction m_function;

    Neuron(int numOutputs, int myIndex, int myLayer, INeuronFunction function, double eta, double alpha)
    {
        this.eta = eta;
        this.alpha = alpha;
        m_myIndex = myIndex;
        m_myLayer = myLayer;

        m_function = function;
        m_outputWeights = new Connection[numOutputs]; // One weight for each of the destination neurons in the next layer
        for (int c = 0; c < numOutputs; c++) {
            m_outputWeights[c] = new Connection();
            m_outputWeights[c].weight = randomWeight();
        }

    }

    void setOutputVal(double val)
    {
        m_outputVal = val;
    }

    double getOutputVal()
    {
        return m_outputVal;
    }

    void feedForward(final Layer prevLayer)
    {
        double sum = 0.0;

        // Sum the previous layer's outputs (which are our inputs)
        // Include the bias node from the previous layer.
        for (int n = 0; n < prevLayer.neurons.length; ++n) {
            sum += prevLayer.neurons[n].getOutputVal() *
                    prevLayer.neurons[n].m_outputWeights[m_myIndex].weight;
        }

        m_outputVal = m_function.transferFunction(sum);
    }

    void calcOutputGradients(double targetVal)
    {
        double delta = targetVal - m_outputVal;
        m_gradient = delta * m_function.transferFunctionDerivative(m_outputVal);
    }

    void calcHiddenGradients(final Layer nextLayer)
    {
        double dow = sumDOW(nextLayer);
        m_gradient = dow * m_function.transferFunctionDerivative(m_outputVal);
    }

    void updateInputWeights(Layer prevLayer)
    {
        // The weights to be updated are in the Connection container
        // in the neurons in the preceding layer
        for (int n = 0; n < prevLayer.neurons.length; ++n) {
            Neuron neuron = prevLayer.neurons[n];
            double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;
            double newDeltaWeight =
                    // Individual input, magnified by the gradient and train rate:
                    eta
                            * neuron.getOutputVal()
                            * m_gradient
                            // Also add momentum = a fraction of the previous delta weight;
                            + alpha
                            * oldDeltaWeight;

            neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
            neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
        }

    }

    private double sumDOW(Layer nextLayer)
    {
        double sum = 0.0;

        // Sum our contributions of the errors at the nodes we feed.
        for (int n = 0; n < nextLayer.neurons.length - 1; ++n)
        {
            sum += m_outputWeights[n].weight * nextLayer.neurons[n].m_gradient;
        }

        return sum;
    }

    private static double randomWeight() {
        return Math.random();   // 0 < .. < 1
    }

    /**
     * Holds weights for each of the outputs
     */
    private class Connection
    {
        double weight;
        double deltaWeight;
        @Override
        public String toString()
        {
            return "(" + weight + "," + deltaWeight + ")";
        }
    }

    @Override
    public String toString()
    {
        StringBuilder b = new StringBuilder();
        b.append("Layer: ").append(m_myLayer).append(" Node: ").append(m_myIndex).append(" Weights(w,d) { ");
        for(Connection c : m_outputWeights)
        {
            b.append(c.toString()).append(" ");
        }
        b.append(" }\n");
        return b.toString();
    }
}
