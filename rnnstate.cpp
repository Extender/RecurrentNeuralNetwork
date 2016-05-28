#include "rnnstate.h"

RNNState::RNNState(RNNState *copyFrom, uint32_t _inputCount, uint32_t _outputCount, uint32_t _layerCount, uint32_t *_layerNeuronCounts)
{
    // Here, the input and output layers are meant to be included in "_layerCount" and "_layerNeuronCounts".
    bool copy=copyFrom!=0;
    if(copy)
    {
        inputCount=copyFrom->inputCount;
        outputCount=copyFrom->outputCount;
        inputAndOutputCount=inputCount+outputCount;
        layerCount=copyFrom->layerCount;
        size_t layerCountArraySize=layerCount*sizeof(uint32_t);
        layerNeuronCounts=(uint32_t*)malloc(layerCountArraySize);
        memcpy(layerNeuronCounts,copyFrom->layerNeuronCounts,layerCountArraySize);
    }
    else
    {
        inputCount=_inputCount;
        outputCount=_outputCount;
        inputAndOutputCount=inputCount+outputCount;
        layerCount=_layerCount;
        size_t layerCountArraySize=layerCount*sizeof(uint32_t);
        layerNeuronCounts=(uint32_t*)malloc(layerCountArraySize);
        memcpy(layerNeuronCounts,_layerNeuronCounts,layerCountArraySize);
    }
    size_t layerCountDoublePointerBasedArraySize=layerCount*sizeof(double*);
    size_t layerCountMinusOneDoublePointerPointerBasedArraySize=(layerCount-1)*sizeof(double**);
    weights=(double***)malloc(layerCountMinusOneDoublePointerPointerBasedArraySize);
    neuronValues=(double**)malloc(layerCountDoublePointerBasedArraySize);
    biasWeights=(double**)malloc(layerCountDoublePointerBasedArraySize);
    input=(double*)malloc(inputCount*sizeof(double));
    uint32_t outputBasedDoubleArraySize=outputCount*sizeof(double);
    output=(double*)malloc(outputBasedDoubleArraySize);
    previousOutput=(double*)malloc(outputBasedDoubleArraySize);

    // Copy or initialize values

    if(copy)
    {
        for(uint32_t thisLayer=0;thisLayer<layerCount;thisLayer++)
        {
            uint32_t neuronsInThisLayer=layerNeuronCounts[thisLayer];
            uint32_t neuronsInPreviousLayer=thisLayer>0?layerNeuronCounts[thisLayer-1]:0;
            // The neuron values do not need to be initialized.
            uint32_t neuronsInThisLayerBasedDoubleArraySize=neuronsInThisLayer*sizeof(double);
            neuronValues[thisLayer]=(double*)malloc(neuronsInThisLayerBasedDoubleArraySize);
            if(thisLayer>0) // The input layer has no bias weights/weights pointing to it.
            {
                uint32_t weightLayerIndex=thisLayer-1 /*Input layer not included*/;

               biasWeights[weightLayerIndex]=(double*)malloc(neuronsInThisLayerBasedDoubleArraySize);
               memcpy(biasWeights[weightLayerIndex],copyFrom->biasWeights[weightLayerIndex],neuronsInThisLayerBasedDoubleArraySize);

               weights[weightLayerIndex]=(double**)malloc(neuronsInPreviousLayer*sizeof(double*));
               for(uint32_t neuronInPreviousLayer=0;neuronInPreviousLayer<neuronsInPreviousLayer;neuronInPreviousLayer++)
               {
                   // Perform deep copy
                   weights[weightLayerIndex][neuronInPreviousLayer]=(double*)malloc(neuronsInThisLayerBasedDoubleArraySize);
                   memcpy(weights[weightLayerIndex][neuronInPreviousLayer],copyFrom->weights[weightLayerIndex][neuronInPreviousLayer],neuronsInThisLayerBasedDoubleArraySize);
               }
            }
        }
    }
    else
    {
        srand(time(0));
        for(uint32_t thisLayer=0;thisLayer<layerCount;thisLayer++)
        {
            uint32_t neuronsInThisLayer=layerNeuronCounts[thisLayer];
            // The neuron values do not need to be initialized.
            uint32_t neuronsInThisLayerBasedDoubleArraySize=neuronsInThisLayer*sizeof(double);
            neuronValues[thisLayer]=(double*)malloc(neuronsInThisLayerBasedDoubleArraySize);
            if(thisLayer>0) // The input layer has no bias weights/weights pointing to it.
            {
               biasWeights[thisLayer-1]=(double*)malloc(neuronsInThisLayerBasedDoubleArraySize);
               for(uint32_t neuronInThisLayer=0;neuronInThisLayer<neuronsInThisLayer;neuronInThisLayer++)
                   biasWeights[thisLayer-1][neuronInThisLayer]=0.0;

                uint32_t neuronsInPreviousLayer=layerNeuronCounts[thisLayer-1];
                uint32_t weightLayerIndex=thisLayer-1 /*Input layer not included*/;
                weights[weightLayerIndex]=(double**)malloc(neuronsInPreviousLayer*sizeof(double*));
                for(uint32_t neuronInPreviousLayer=0;neuronInPreviousLayer<neuronsInPreviousLayer;neuronInPreviousLayer++)
                {
                    weights[weightLayerIndex][neuronInPreviousLayer]=(double*)malloc(neuronsInThisLayerBasedDoubleArraySize);
                    for(uint32_t neuronInThisLayer=0;neuronInThisLayer<neuronsInThisLayer;neuronInThisLayer++)
                        weights[weightLayerIndex][neuronInPreviousLayer][neuronInThisLayer]=-0.1+(((double)rand())/((double)RAND_MAX))*0.2;
                }
            }
        }
    }
}

RNNState::~RNNState()
{
    free(input);
    free(output);
    free(previousOutput);
    for(uint32_t thisLayer=0;thisLayer<layerCount;thisLayer++)
    {
        free(neuronValues[thisLayer]);
        if(thisLayer>0) // The input layer has no bias weights/weights pointing to its neurons.
        {
            uint32_t neuronsInPreviousLayer=layerNeuronCounts[thisLayer-1 /*Neuron count of previous layer*/];
            uint32_t weightLayerIndex=thisLayer-1 /*Input layer not included*/;
            free(biasWeights[weightLayerIndex]);
            for(uint32_t neuronInPreviousLayer=0;neuronInPreviousLayer<neuronsInPreviousLayer;neuronInPreviousLayer++)
                free(weights[weightLayerIndex][neuronInPreviousLayer]);
            free(weights[weightLayerIndex]);
        }
    }
    free(neuronValues);
    free(biasWeights);
    free(weights);
    free(layerNeuronCounts);
}
