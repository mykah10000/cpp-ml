#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <random>
using namespace std;

struct node {
    vector<float> weights;
    float bias;
    float output;
};

struct network{
    private:
        vector<layer> layers;

    public:
        void calculate_value(){
            for(int i = 1; i<layers.size();i++){
                for(int p = 0; p<layers[i].checkSize();p++){
                    for(int k = 0; k<layers[i-1].checkSize();k++){
                        output[i] += (currentLayer.weights[i][k]*prevLayer.output[k]);
                        cout << "Neuron: " << i << ". Weight * prevLayer output " << k << ": " << currentLayer.weights[i][k]*prevLayer.output[k] << "\n";
                    }
                output[i] += currentLayer.biases[i];
                }
            }
        }
};

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

        void set_weights_rand(layer prevLayer){
        int num_neurons = biases.size();
        int num_inputs = prevLayer.output.size();
        vector<vector<float>> output(num_neurons, vector<float>(num_inputs));
        random_device dev;
        mt19937 rng(dev());
        uniform_real_distribution<float> dist(0.0f,1.0f);
        for(int i = 0; i<biases.size();i++){
            for(int k = 0; k<prevLayer.output.size();k++){
                weights[i][k] = dist(rng);
                cout << "Neuron: " << i << ". Weight: " << k << ": " << output[i][k] << "\n";
            }
        }
        }
        int checkSize(){
            return output.size();
        }

        
};





// vector<layer> take_values(string file_name){
//     fstream values(file_name);
// }