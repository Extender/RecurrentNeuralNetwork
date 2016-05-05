#ifndef RNNSTATE_H
#define RNNSTATE_H

#include <stdint.h>
#include <stdlib.h>
#include <string>
#include <ctime>

// Here, the input and output layers are meant to be included in "_layerCount" and "_layerNeuronCounts".

class RNNState
{
public:
    // Dimensions: layers -> neurons in this layer -> neurons in next layer
    double ***weights;

    // Dimensions: layers -> neuron values / neuron bias weights
    double **neuronValues;
    double **biasWeights;

    uint32_t *layerNeuronCounts;
    uint32_t layerCount;

    double *input;
    double *previousOutput;
    double *output;

    uint32_t inputCount;
    uint32_t outputCount;
    uint32_t inputAndOutputCount;


public:
    RNNState(RNNState *copyFrom,uint32_t _inputCount=0,uint32_t _outputCount=0,uint32_t _layerCount=0,uint32_t *_layerNeuronCounts=0);
    ~RNNState();
};

#endif // RNNSTATE_H
