package test;

import neuralnet.*;

import java.util.Arrays;

/**
 * Test run of the neural net
 */
public class NetTest
{
    static private void log(String msg)
    {
        System.err.println(msg);
    }

    public static void main(String args[])
    {
        ITrainingData trainData = new XORTrainingData();

        // e.g., { 3, 2, 1 }
        int[] topology = trainData.getTopology();

        INeuralNet myNet = NeuralNetFactory.CreateNetwork(topology, new TanHNeuronFunction(), 1.0, 0.5, 0.2);

        double[] inputVals;
        double [] targetVals;
        double [] resultVals;

        int trainingPass = 0;

        while (!trainData.isEof()) {
            ++trainingPass;
            log("Pass " + trainingPass);

            // Get new input data and feed it forward:
            inputVals = trainData.getNextInputs();
            targetVals = trainData.getTargetOutputs();

            if (inputVals.length != topology[0]) {
                break;
            }

            log("Inputs:" + Arrays.toString(inputVals));
            resultVals = myNet.feedForward(inputVals);

            // Collect the net's actual output results (returned from the feedForward run).
            log("Outputs:" + Arrays.toString(resultVals));

            // Train the net what the outputs should have been:
            log("Targets:" + Arrays.toString(targetVals));
            myNet.backProp(targetVals);

            // Report how well the training is working, average over recent samples:
            log("Net recent average error: "+myNet.getRecentAverageError());
        }
        log("Done");
        log(myNet.toString());
    }
}
