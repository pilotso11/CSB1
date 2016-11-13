package neuralnet;

/**
 * Rectifier implementation
 * See
 *      https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
 */
public class RectifierNeuronFunction implements INeuronFunction
{
    @Override
    public double transferFunction(double x)
    {
        // rectifier is max(0, x)
        return Math.max(0, x);
    }

    @Override
    public double transferFunctionDerivative(double x)
    {
        //(d/dx)(max(0,x)) = 1 if x > 0, otherwise 0
        return x > 0 ? 1 : 0;
    }
}

