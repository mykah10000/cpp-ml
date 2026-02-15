#include "ml.h"
#include <vector>
#include <string>

using namespace std;

int main(){
    layer layer0(2,0);
    layer0.output = {.5,.1};
    layer layer1;
    layer1.biases = {.6,.3,.9};
    layer layer2;
    layer2.biases = {.2};
    layer1.weights = layer1.set_weights_rand(layer1,layer0);
    layer1.output = layer1.calculate_value(layer1, layer0);
    layer2.weights = layer2.set_weights_rand(layer2, layer1);
    layer2.output = layer2.calculate_value(layer2, layer1);
    for(float i : layer1.output){
        cout << i << "\n";
    }
    cout << "Final output: " << layer2.output[0];
    return 0;
}
