#include "ml.h"

int main(){
    vector<int> networkNodes = {2,2,2,1};
    int numLayers = 4;
    network nn;
    nn.create(numLayers, networkNodes);
    nn.display();

    vector<double> inputs = {1,0};
    vector<vector<double>> inputVector = {inputs};
    vector<double> y = {1};
    // nn.calculate(inputs);
    nn.backprop(inputVector, y, .001);
}