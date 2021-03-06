// Alexey Gavryushin / 2016; this code is hereby released into the public domain.
// For suggestions, please contact int01@outlook.com

#include <stdlib.h>
#include <iostream>
#include <stdint.h>
#include <limits>

#include "io.h"
#include "text.h"

#include "rnn.h"

using namespace std;

string doubleArrayToString(double *array,uint32_t elementCount,bool includeHighest)
{
    string out;
    out+="[";
    double highest=std::numeric_limits<double>::min();
    uint32_t highestIndex;
    for(uint32_t i=0;i<elementCount;i++)
    {
        if(i>0)
            out+=", ";
        char *str=text::doubleToString(array[i]);
        if(includeHighest&&array[i]>highest)
        {
            highest=array[i];
            highestIndex=i;
        }
        out+=str;
        free(str);
    }
    if(includeHighest&&elementCount>0)
    {
        out+=" (highest: ";
        char *highestStr=text::unsignedIntToString(highestIndex);
        out+=highestStr;
        out+=")";
        free(highestStr);
    }
    out+="]";
    return out;
}

int main(int argc, char *argv[])
{
    /*
    This implementation is a Jordan-type recurrent neural network.
    A computational step takes the current input data plus the output data of the last computational step (or zeroes, if it is the first step).
    */

    const char *helloString="hello";
    uint32_t inputCount=3; // h, e, l
    uint32_t outputCount=3; // e, l, o
    uint32_t additionalMemoryNeuronCount=3; // These neurons help the RNN by effectively turning into adjustable parameters during training
    uint32_t effectiveOutputCount=outputCount+additionalMemoryNeuronCount;
    uint32_t backpropagationSteps=3;
    double learningRate=0.1;
    double momentum=0.9;
    double weightDecay=0.0001;
    RNN *rnn=new RNN(inputCount,effectiveOutputCount,backpropagationSteps,learningRate,momentum,weightDecay,2);
    uint64_t cycle=0;
    char *str;
    double **desiredOutputs=(double**)malloc((backpropagationSteps+1)*sizeof(double*));
    // Only call learn() after the last step!

    vector<int> accuracyVector;
    double accuracySum=0.0;

    for(uint64_t current=0;/*current<10*/;current++)
    {
        double *input=(double*)malloc(inputCount*sizeof(double));
        uint64_t currentPos=current%4; // o not used!
        if(currentPos==0)
        {
            str=text::unsignedLongToString(cycle);
            cout<<"*** New cycle (#"<<str<<") ***"<<endl<<endl;
            free(str);
        }
        uint8_t currentChar;
        if(currentPos==0)
            currentChar=0;
        else if(currentPos==1)
            currentChar=1;
        else // if(currentPos==2||currentPos==3)
            currentChar=2;
        str=text::unsignedLongToString(currentPos);
        cout<<"Current position: "<<str<<" ("<<helloString[currentPos]<<")"<<endl;
        free(str);
        for(uint32_t i=0;i<inputCount;i++)
            input[i]=(i==currentChar?1.0:0.0);
        double *output;
        output=rnn->process(input);
        cout<<"Output:           "<<doubleArrayToString(output,outputCount /*Do not include the additional memory neurons*/,true)<<endl;

        double *desiredOutput=(double*)malloc(effectiveOutputCount*sizeof(double));
        // Desired output: next char!
        uint8_t desiredOut;
        if(currentPos==0) // "h"
            desiredOut=0; // => e
        else if(currentPos==1) // "e"
            desiredOut=1; // => l
        else if(currentPos==2) // "l"
            desiredOut=1; // => l
        else // if(currentPos==3) // "l"
            desiredOut=2; // => o
        for(uint32_t i=0;i<effectiveOutputCount;i++)
            desiredOutput[i]=(i==desiredOut?1.0:(i>=outputCount?output[i]:0.0 /*Do not indicate an error if this is an additional memory neuron*/));

        cout<<"Desired output:   "<<doubleArrayToString(desiredOutput,outputCount /*Do not include the additional memory neurons*/,false)<<endl;

        desiredOutputs[currentPos]=desiredOutput;

        uint8_t highestIndex=255;
        double highestValue=std::numeric_limits<double>::min();
        for(uint8_t i=0;i<effectiveOutputCount;i++)
        {
            if(output[i]>highestValue) // Not newOutput!
            {
                highestIndex=i;
                highestValue=output[i];
            }
        }

        bool wasCorrect=(highestIndex==desiredOut);
        int avSize=accuracyVector.size();
        if(avSize>=100)
        {
            accuracySum-=accuracyVector.at(0);
            accuracyVector.erase(accuracyVector.begin());
        }
        else
            avSize++;

        accuracyVector.push_back(wasCorrect);
        accuracySum+=wasCorrect?1.0:0.0;

        str=text::doubleToStringWithFixedPrecision((accuracySum/((double)avSize))*100.0,0);
        cout<<"Accuracy of last 100 outputs :    "<<str<<"%"<<endl;
        free(str);

        if(currentPos==3)
        {
            rnn->learn(desiredOutputs);
            free(desiredOutputs[0]);
            free(desiredOutputs[1]);
            free(desiredOutputs[2]);
            free(desiredOutputs[3]);
            cycle++;
        }
        free(output);
        free(input);

        cout<<endl;
    }
    free(desiredOutputs);
    delete rnn;
}

