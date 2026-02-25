#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <random>
using namespace std;

random_device rd;
mt19937 gen(rd());

struct node {
    private:
        vector<double> weights;
        double bias = 0;
        double activation = 0;

    public:
        void create(int prevLen,const vector<double>& w, double b){
            if(prevLen<1){
                cout << "Node creation: prev layer length has to be larger than 0\n";
                return;
            }
            weights = vector<double>(prevLen);
            for(int i = 0;i<prevLen;i++){
                weights[i] = w[i];
            }
            bias = b;
            cout << "Node creation: successfully created node\n";
        }

        void create(int prevLen){
            if(prevLen<1){
                cout << "Node creation 1: prev layer length has to be larger than 0\n";
                return;
            }
            uniform_real_distribution<> randWeight(-1,1);
            weights = vector<double>(prevLen);
            for(int i = 0; i<prevLen;i++){
                weights[i] = randWeight(gen);
            }
            bias = randWeight(gen);
            cout << "Node Creation: successfully created node\n";
        }

        void calculate(const vector<node>& inputs){
            if(inputs.size() != weights.size()){
                cout << "Node calculation: Activations have to have as many variables as node has weights\n";
                return;
            }
            if(weights.size() == 0){
                cout << "Node calculation: Weights not iniitialized, 0 Weights.\n";
                return;
            }
            double sum = bias;
            for(int i = 0;i<inputs.size();i++){
                sum += inputs[i].activation * weights[i];
            }
            activation = 1/(1+exp(-sum));
        }

        void setValues(vector<double>& w, double b){
            for(int i = 0;i<w.size();i++){
                // cout << "Old Weight: " << weights[i] << ", New weight: " << weights[i]-w[i];
                weights[i] -= w[i];
            }
            bias -= b;
        }

        void inputLayerCreate(double actValue){
            activation = actValue;
        }

        double returnActivation(){
            return activation;
        }

        vector<double> returnWeights(){
            return weights;
        }
};

struct layer{
    private:
        vector<node> nodes;

    public:
        void create(int numNodes, int numInputs){
            if(numNodes <= 0 || numInputs <= 0){
                cout << "Layer creation: needs more than 1 node and input\n";\
                return;
            }
            nodes = vector<node>(numNodes);
            for(int i = 0; i<numNodes; i++){
                nodes[i].create(numInputs);
            }
        }

        void firstLayerCreate(const vector<double>& inputs){
            if(inputs.size() <= 0){
                cout << "Input layer creation: needs more than 0 nodes\n";
                return;
            }
            nodes = vector<node>(inputs.size());
            for(int i = 0; i<inputs.size();i++){
                nodes[i].inputLayerCreate(inputs[i]);
            }
        }

        void calculate(const layer& inputs){
            if(inputs.nodes.size() < 1){
                cout << "layer calculation: previous layer has to be larger than 0\n";
                return;
            }
            if(nodes.size() < 1){
                cout << "layer calculation: layer not initialized, no nodes exist in layer\n";
            }
            for(int i = 0; i < nodes.size(); i++){
                nodes[i].calculate(inputs.nodes);
            }
        }

        int size(){
            return nodes.size();
        }

        double activation(int i){
                // cout << "Node: " << i << ", Activation: " << nodes[i].returnActivation() << "\n";
                return nodes[i].returnActivation();
        }

        vector<double> weights(int i){
            return nodes[i].returnWeights();
        }

        void setValues(vector<double>& w, double b, int n){
            nodes[n].setValues(w, b);
        }
};

struct network{
    private:
        vector<layer> layers;
        double y_hat;
        vector<vector<double>> S;

    public:
        void create(int numLayers, const vector<int>& numNodes){
            if(numLayers < 2){
                cout << "needs to be at least 2 layers\n";
                return;
            }
            layers = vector<layer>(numLayers);
            S = vector<vector<double>>(numLayers-1);
            for(int i = numLayers-2; i >= 0; i--){
                S[i] = vector<double>(numNodes[i+1]);
            }
            layers[0].firstLayerCreate(vector<double>(numNodes[0]));
            for(int i = 1; i<numLayers; i++){
                layers[i].create(numNodes[i], numNodes[i-1]);
            }
        }

        void display(){
            for(int i = 0; i<layers.size();i++){
                cout << "Layer: " << i;
                for(int k = 0;k<layers[i].size();k++){
                    cout << ", " << k;
                }
                cout << "\n";
            }
        }

        void calculate(const vector<double>& inputs){
            if(inputs.size() != layers[0].size()){
                cout << "Network calcuation: incorrect input size\n";
                return;
            }
            layers[0].firstLayerCreate(inputs);
            for(int i = 1; i<layers.size();i++){
                layers[i].calculate(layers[i-1]);
            }
            y_hat = layers[layers.size()-1].activation(0);
            cout << "Y_hat: " << y_hat << "\n";
        }

        void backprop(const vector<vector<double>>& x, const vector<double>& y, double lr){
            if(y.size() <1 || y.size() <1){
                cout << "Backprop: invalid inputs\n";
                return;
            }
            for(int i = 0; i < y.size();i++){
                calculate(x[i]);
                S[S.size()-1][0] = 2*(y_hat-y[i])*(layers[layers.size()-1].activation(0)*(1-layers[layers.size()-1].activation(0)));
                // S[S.size()-1][0] = (y_hat-y[i]);
                cout << "Backprop S[" << S.size()-1 << "][0]: " << S[S.size()-1][0] << "\n";

                
                for(int j = S.size()-2; j>=0;j--){
                    for(int k = 0;k<S[j].size();k++){
                        double summation = 0;
                        double dSigmoid = (layers[j+1].activation(k)*(1-layers[j+1].activation(k)));
                        for(int p = 0;p<layers[j+2].size();p++){
                            summation += (layers[j+2].weights(p)[k]*S[j+1][p]);
                        }
                        S[j][k] = summation * dSigmoid;
                        cout << "backprop S[" << j << "][" << k << "]: " << S[j][k] << "\n";
                    }
                }


                for(int j = S.size()-1; j>=0;j--){
                    for(int k = 0;k<S[j].size();k++){
                        vector<double> weights;
                        for(int p = 0;p<layers[j].size();p++){
                            weights.push_back(lr*S[j][k]*layers[j].activation(p));
                        }
                        double bias = lr * S[j][k];
                        layers[j+1].setValues(weights,bias,k);
                    }
                }
            }
        }
};