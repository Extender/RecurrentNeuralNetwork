#include "rnn.h"

double RNN::sig(double input)
{
    // Derivative: sig(input)*(1.0-sig(input))
    return 1.0/(1.0+pow(M_E,-input));
}

double RNN::tanh(double input)
{
    // Derivative: 1.0-((tanh(input))^2)
    return (1.0-pow(M_E,-2.0*input))/(1.0+pow(M_E,-2.0*input));
}

RNN::RNN(uint32_t _inputCount, uint32_t _outputCount, uint32_t _backpropagationSteps, double _learningRate, uint32_t _layerCount, uint32_t *_layerNeuronCounts)
{
    // Here, the input and output layers are meant to be included in "_layerCount" and "_layerNeuronCounts".
    inputCount=_inputCount;
    outputCount=_outputCount;
    inputAndOutputCount=inputCount+outputCount;
    backpropagationSteps=_backpropagationSteps;
    learningRate=_learningRate;
    layerCount=_layerCount;
    if(_layerCount<2)
        throw;

    layerNeuronCounts=(uint32_t*)malloc(layerCount*sizeof(uint32_t));
    if(_layerNeuronCounts==0)
    {
        for(uint32_t thisLayer=0;thisLayer<layerCount;thisLayer++)
            layerNeuronCounts[thisLayer]=(thisLayer==0?inputAndOutputCount /*Input and previous output*/:(thisLayer==layerCount-1?outputCount:inputAndOutputCount));
    }
    else
    {
        memcpy(layerNeuronCounts,_layerNeuronCounts,layerCount*sizeof(double));
        layerNeuronCounts[0]=inputAndOutputCount; // Must have this size.
    }

    stateArraySize=2*backpropagationSteps+1 /*One for the current state.*/;
    stateArrayPos=0xffffffff;
    states=(RNNState**)malloc(stateArraySize*sizeof(RNNState*));
}

RNN::~RNN()
{
    for(uint32_t layer=stateArrayPos-backpropagationSteps;layer<=stateArrayPos;layer++)
        delete states[layer];
    free(states);
    free(layerNeuronCounts);
}

RNNState *RNN::pushState()
{
    // This works as follows: the buffer is larger (usually 2 times larger) than the required size, allowing us to avoid having to move memory
    // every time a new state is pushed. Once the buffer is filled, the needed elements in the front are moved back, overriding the old states
    // that aren't needed anymore, and creating room for new states to be pushed.

    if(stateArrayPos==0xffffffff)
        stateArrayPos=0; // Do not increment the position the first time pushLayerState() is called.
    else
    {
        if(stateArrayPos==stateArraySize-1)
        {
            // Overwrite old states that aren't needed anymore, and set the new position:
            // Note that the current state will be a backpropagation state after the new state is pushed to the array.
            delete states[stateArrayPos-backpropagationSteps]; // Delete unneeded state
            memcpy(states,states+(stateArraySize-backpropagationSteps),backpropagationSteps*sizeof(RNNState*));
            stateArrayPos=backpropagationSteps-1;
        }
        stateArrayPos++;
    }
    // Copy values from previous state, if such a state exists:
    RNNState *newState=stateArrayPos>0/*Has previous state?*/?new RNNState(getState(1)):new RNNState(0,inputCount,outputCount,layerCount,layerNeuronCounts);
    states[stateArrayPos]=newState;
    if(stateArrayPos>backpropagationSteps)
    {
        // Free memory occupied by the now unneeded state (each time a new state is pushed, the memory occupied by the oldest state, which is
        // not needed anymore from that point on, is freed):
        delete states[stateArrayPos-backpropagationSteps-1];
    }
    return states[stateArrayPos];
}

RNNState *RNN::getCurrentState()
{
    return states[stateArrayPos];
}

bool RNN::hasState(uint32_t stepsBack)
{
    return stateArrayPos!=0xffffffff&&stepsBack<=__min(backpropagationSteps,stateArrayPos);
}

uint32_t RNN::getAvailableStepsBack()
{
    return stateArrayPos!=0xffffffff?__min(backpropagationSteps,stateArrayPos):0;
}

RNNState *RNN::getState(uint32_t stepsBack)
{
    return states[stateArrayPos-stepsBack];
}

double *RNN::process(double *input)
{
    // Effective input: input plus previous output.
    RNNState *newState=pushState();
    bool hasPreviousState=hasState(1);
    RNNState *previousState=hasPreviousState?getState(1):0;
    uint32_t inputCountBasedDoubleArraySize=inputCount*sizeof(double);
    uint32_t outputCountBasedDoubleArraySize=outputCount*sizeof(double);
    memcpy(newState->input,input,inputCountBasedDoubleArraySize);
    memcpy(newState->neuronValues[0],input,inputCountBasedDoubleArraySize);
    if(hasPreviousState)
    {
        memcpy(newState->previousOutput,previousState->output,outputCountBasedDoubleArraySize);
        memcpy(newState->neuronValues[0]+inputCount,previousState->output,outputCountBasedDoubleArraySize);
    }
    else
    {
        // Initialize neuron values with zeroes (needed).
        for(uint32_t i=0;i<outputCount;i++)
        {
            newState->previousOutput[i]=0.0;
            newState->neuronValues[0][inputCount+i]=0.0;
        }
    }
    double *output=(double*)malloc(outputCountBasedDoubleArraySize);
    for(uint32_t thisLayer=0;thisLayer<layerCount-1 /*Do not include output layer*/;thisLayer++)
    {
        uint32_t neuronsInThisLayer=layerNeuronCounts[thisLayer];
        uint32_t neuronsInNextLayer=layerNeuronCounts[thisLayer+1];

        for(uint32_t neuronInNextLayer=0;neuronInNextLayer<neuronsInNextLayer;neuronInNextLayer++)
        {
            double thisLayerNeuronValueMultipliedByWeightSum=0.0;
            for(uint32_t neuronInThisLayer=0;neuronInThisLayer<neuronsInThisLayer;neuronInThisLayer++)
            {
                double valueOfNeuronInThisLayer=newState->neuronValues[thisLayer][neuronInThisLayer];
                thisLayerNeuronValueMultipliedByWeightSum+=newState->weights[thisLayer /*Input layer not included; effectively thisLayer-1+1*/][neuronInThisLayer][neuronInNextLayer]*valueOfNeuronInThisLayer;
            }
            double newNeuronValue=tanh(thisLayerNeuronValueMultipliedByWeightSum+newState->biasWeights[thisLayer/*The input layer has no bias weights, so this is thisLayer+1-1*/][neuronInNextLayer]);
            newState->neuronValues[thisLayer+1][neuronInNextLayer]=newNeuronValue;
        }
    }

    memcpy(newState->output,newState->neuronValues[layerCount-1],outputCountBasedDoubleArraySize);
    memcpy(output,newState->neuronValues[layerCount-1],outputCountBasedDoubleArraySize);
    return output;
}

void RNN::learn(double **desiredOutputs)
{
    uint32_t availableStepsBack=getAvailableStepsBack();
    // Dimensions: layers -> neurons in this layer -> neurons in next layer
    double ***weightDiff=(double***)malloc((layerCount-1)*sizeof(double**));
    // Dimensions: layers -> neurons in layer
    double **biasDiff=(double**)malloc(layerCount*sizeof(double*));

    double *bottomDiff=(double*)malloc(outputCount*sizeof(double)); // Derivatives of the loss function w.r.t. the previous outputs; does not need to be initialized.

    RNNState *latestState=getCurrentState();
    bool weightsAllocated=false;

    // This will cycle totalStepCount times, but we need to go backwards, so we use "stepsBack" in combination with "getState(stepsBack)".

    for(uint32_t stepsBack=0;stepsBack<=availableStepsBack;stepsBack++)
    {
        // 0 = current state
        RNNState *thisState=getState(stepsBack);
        double **errorTerms=(double**)malloc((layerCount-1)*sizeof(double*)); // The input layer has no error terms.

        for(uint32_t thisLayer=layerCount-1;thisLayer>0;thisLayer--) // Input layer not included.
        {
            uint32_t neuronsInThisLayer=layerNeuronCounts[thisLayer];
            uint32_t neuronsInPreviousLayer=layerNeuronCounts[thisLayer-1];
            uint32_t neuronsInNextLayer=thisLayer==layerCount-1?0:layerNeuronCounts[thisLayer+1];
            errorTerms[thisLayer-1 /*Input layer not included*/]=(double*)malloc(neuronsInThisLayer*sizeof(double));
            if(!weightsAllocated)
            {
                weightDiff[thisLayer-1 /*Input layer not included*/]=(double**)malloc(neuronsInPreviousLayer*sizeof(double*));
                biasDiff[thisLayer-1 /*Input layer not included*/]=(double*)malloc(neuronsInThisLayer*sizeof(double));
            }
            for(uint32_t neuronInThisLayer=0;neuronInThisLayer<neuronsInThisLayer;neuronInThisLayer++)
            {
                // Calculate the derivative of the loss function w.r.t the value inside the tanh function of this neuron ("error term"):

                double outputValue=thisState->neuronValues[thisLayer][neuronInThisLayer]; // Output value of this neuron
                double sumOfErrorTermsMultipliedByWeightsInNextLayer=0.0;
                if(!weightsAllocated)
                    biasDiff[thisLayer-1 /*Input layer not included*/][neuronInThisLayer]=0.0;
                for(uint32_t neuronInNextLayer=0;neuronInNextLayer<neuronsInNextLayer;neuronInNextLayer++)
                {
                    double errorTermOfNeuronInNextLayer=errorTerms[thisLayer /*Input layer not included; effectively thisLayer-1+1*/][neuronInNextLayer];
                    sumOfErrorTermsMultipliedByWeightsInNextLayer+=thisState->weights[thisLayer /*Input layer not included; effectively thisLayer-1+1*/][neuronInThisLayer][neuronInNextLayer]*errorTermOfNeuronInNextLayer;
                }
                double errorTerm;
                if(thisLayer==layerCount-1)
                {
                    // Bottom diff value: derivative of the loss function w.r.t. the value of this neuron
                    double bottomDiffValue=(stepsBack>0?bottomDiff[neuronInThisLayer]:0.0);
                    errorTerm=(1.0-pow(outputValue,2))*((desiredOutputs[availableStepsBack-stepsBack][neuronInThisLayer]-outputValue)+bottomDiffValue);
                }
                else
                    errorTerm=(1.0-pow(outputValue,2))*sumOfErrorTermsMultipliedByWeightsInNextLayer;
                errorTerms[thisLayer-1 /*Input layer not included*/][neuronInThisLayer]=errorTerm;

                // Update weights pointing to this neuron:

                for(uint32_t neuronInPreviousLayer=0;neuronInPreviousLayer<neuronsInPreviousLayer;neuronInPreviousLayer++)
                {
                    if(!weightsAllocated)
                    {
                        if(neuronInThisLayer==0) // Allocate weight diffs of neurons in previous layer upon first iteration of "neuronInThisLayer".
                            weightDiff[thisLayer-1 /*Input layer not included*/][neuronInPreviousLayer]=(double*)malloc(neuronsInThisLayer*sizeof(double));
                        weightDiff[thisLayer-1 /*Input layer not included*/][neuronInPreviousLayer][neuronInThisLayer]=0.0;
                    }
                    double valueOfNeuronInPreviousLayer=thisState->neuronValues[thisLayer-1][neuronInPreviousLayer];
                    weightDiff[thisLayer-1 /*Input layer not included*/][neuronInPreviousLayer][neuronInThisLayer]+=errorTerm*valueOfNeuronInPreviousLayer;
                }
                // Update bias
                if(!weightsAllocated)
                    biasDiff[thisLayer-1 /*Input layer not included*/][neuronInThisLayer]=0.0;
                biasDiff[thisLayer-1 /*Input layer not included*/][neuronInThisLayer]+=errorTerm;
            }
            if(thisLayer<layerCount-1)
                free(errorTerms[thisLayer /*Input layer not included; effectively thisLayer-1+1*/]);
        }

        // Calculate bottomDiff:

        for(uint32_t previousOutputInputNeuron=0;previousOutputInputNeuron<outputCount;previousOutputInputNeuron++)
        {
            double errorTermTimesWeightSum=0.0;
            uint32_t neuronsInNextLayer=layerNeuronCounts[1];
            for(uint32_t neuronInNextLayer=0;neuronInNextLayer<neuronsInNextLayer;neuronInNextLayer++)
                errorTermTimesWeightSum+=errorTerms[0][neuronInNextLayer]*thisState->weights[0][inputCount+previousOutputInputNeuron][neuronInNextLayer];

            bottomDiff[previousOutputInputNeuron]=errorTermTimesWeightSum;
        }

        free(errorTerms[0]);
        free(errorTerms);

        if(!weightsAllocated)
            weightsAllocated=true;
    }
    free(bottomDiff);

    // Now that we have cycled through all states, apply all changes:

    for(uint32_t thisLayer=layerCount-1;thisLayer>0;thisLayer--) // Input layer not included.
    {
        uint32_t neuronsInThisLayer=layerNeuronCounts[thisLayer];
        uint32_t neuronsInPreviousLayer=layerNeuronCounts[thisLayer-1];
        for(uint32_t neuronInThisLayer=0;neuronInThisLayer<neuronsInThisLayer;neuronInThisLayer++)
        {
            // +=, not -= needed!
            for(uint32_t neuronInPreviousLayer=0;neuronInPreviousLayer<neuronsInPreviousLayer;neuronInPreviousLayer++)
                latestState->weights[thisLayer-1 /*Input layer not included*/][neuronInPreviousLayer][neuronInThisLayer]+=learningRate*weightDiff[thisLayer-1 /*Input layer not included*/][neuronInPreviousLayer][neuronInThisLayer];
            latestState->biasWeights[thisLayer-1 /*Input layer not included*/][neuronInThisLayer]+=learningRate*biasDiff[thisLayer-1 /*Input layer not included*/][neuronInThisLayer];
        }
        for(uint32_t neuronInPreviousLayer=0;neuronInPreviousLayer<neuronsInPreviousLayer;neuronInPreviousLayer++)
            free(weightDiff[thisLayer-1][neuronInPreviousLayer]);
        free(biasDiff[thisLayer-1 /*Input layer not included*/]);
        free(weightDiff[thisLayer-1 /*Input layer not included*/]);
    }

    free(biasDiff);
    free(weightDiff);
}
