package neuralnet;

/**
 * TANH neuron implementation.
 */
public class TanHNeuronFunction implements INeuronFunction
{
    @Override
    public double transferFunction(double x)
    {
        // tanh - output range [-1.0..1.0]
        return Math.tanh(x);
    }

    @Override
    public double transferFunctionDerivative(double x)
    {
        // tanh derivative
        return 1.0 - x * x;
    }
}
