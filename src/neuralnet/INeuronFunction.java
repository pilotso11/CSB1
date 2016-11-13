package neuralnet;

/**
 * Neuron Function Interface
 */
public interface INeuronFunction
{
    /**
     *
     * @param x
     * @return
     */
    double transferFunction(double x);

    /**
     *
     * @param x
     * @return
     */
    double transferFunctionDerivative(double x);
}

