#include "ml.h"

int main(){
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> randInt(0,1);
    uniform_real_distribution<> randWeight(-.1,.1);
    vector<vector<double>> input;
    input = vector<vector<double>>(10000);
    vector<double> y;
    y = vector<double>(10000);
    for(int i = 0; i<10000; i++){
        input[i] = vector<double>(2);
        double x1 = static_cast<double>(randInt(gen));
        double x2 = static_cast<double>(randInt(gen));
        input[i][0] = x1;
        input[i][1] = x2;
        y[i] = ((static_cast<int>(input[i][0])+static_cast<int>(input[i][1]))%2);
        // cout << boolalpha << input[i][0] << input[i][1] << " " << y[i] << "\n";
    }




    vector<int> networkNodes = {2,2,1};
    int numLayers = 3;
    network nn;
    nn.create(numLayers, networkNodes);
    nn.display();


    // nn.calculate(inputs);
    nn.backprop(input, y, .1);
}