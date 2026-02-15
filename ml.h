#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <random>
using namespace std;

// struct node {
//     std::vector<float> weights;
//     float bias;
//     float output;
// };

struct layer{
    vector<vector<float>> weights;
    vector<float> biases;
    vector<float> output;
    float cost;
        
    public:
        layer(){}

        layer(int num_neurons, int num_inputs){
            weights.resize(num_neurons, vector<float>(num_inputs));
            biases.resize(num_neurons);
            output.resize(num_neurons);
        }

        vector<vector<float>> set_weights_rand(layer currentLayer, layer prevLayer){
        int num_neurons = currentLayer.biases.size();
        int num_inputs = prevLayer.output.size();
        vector<vector<float>> output(num_neurons, vector<float>(num_inputs));
        random_device dev;
        mt19937 rng(dev());
        uniform_real_distribution<float> dist(0.0f,1.0f);
        for(int i = 0; i<currentLayer.biases.size();i++){
            for(int k = 0; k<prevLayer.output.size();k++){
                output[i][k] = dist(rng);
                cout << "Neuron: " << i << ". Weight: " << k << ": " << output[i][k] << "\n";
            }
        }
        return output; 
        }

        vector<float> calculate_value(layer currentLayer, layer prevLayer){
        vector<float> output(currentLayer.biases.size());
        for(int i = 0; i<currentLayer.biases.size();i++){
            for(int k = 0; k<prevLayer.output.size();k++){
                output[i] += (currentLayer.weights[i][k]*prevLayer.output[k]);
                cout << "Neuron: " << i << ". Weight * prevLayer output " << k << ": " << currentLayer.weights[i][k]*prevLayer.output[k] << "\n";
            }
            output[i] += currentLayer.biases[i];
        }
        return output;
}
};





// vector<layer> take_values(string file_name){
//     fstream values(file_name);
// }