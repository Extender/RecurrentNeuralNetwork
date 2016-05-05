#ifndef RNN_H
#define RNN_H

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

#include <stdint.h>
#include <math.h>

#include "rnnstate.h"


#include <iostream>
#include "text.h"

// Here, the input and output layers are meant to be included in "_layerCount" and "_layerNeuronCounts".

class RNN
{
public:
    uint32_t stateArrayPos;
    uint32_t stateArraySize;
    RNNState **states; // Stores previous iterations

    double learningRate;
    uint32_t inputCount;
    uint32_t outputCount;
    uint32_t inputAndOutputCount;
    uint32_t layerCount;
    uint32_t backpropagationSteps;
    uint32_t *layerNeuronCounts;


    static double sig(double input); // sigmoid function
    static double tanh(double input); // tanh function


    RNNState *pushState();
    RNNState *getCurrentState();
    bool hasState(uint32_t stepsBack);
    uint32_t getAvailableStepsBack();
    RNNState *getState(uint32_t stepsBack);

    RNN(uint32_t _inputCount,uint32_t _outputCount,uint32_t _backpropagationSteps,double _learningRate,uint32_t _layerCount=2,uint32_t *_layerNeuronCounts=0);
    ~RNN();

    double *process(double *input);
    void learn(double **desiredOutputs);
};

#endif // RNN_H
