package test;

import neuralnet.ITrainingData;

/**
 * Create 2000 passes of xor test data
 */

public class XORTrainingData implements ITrainingData
{
    static final int[] topology = {2, 3, 3, 1};
    int loops = 2000;
    int nextOutput;

    @Override
    public int[] getTopology()
    {
        return topology;
    }

    @Override
    public boolean isEof()
    {
        return loops-- == 0;
    }

    @Override
    public double[] getNextInputs()
    {
        int x = loops % 2;
        int y = (loops / 2) % 2;

        nextOutput =  x ^ y ;

        return new double[] {x,y};
    }

    @Override
    public double[] getTargetOutputs()
    {
        return new double[] { nextOutput };
    }
}
