#include "ml.h"

int main(){
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> randInt(0,1);
    uniform_real_distribution<> randWeight(-.1,.1);
    vector<vector<double>> input;
    int n = 20000;
    input = vector<vector<double>>(n);
    vector<vector<double>> y;
    y = vector<vector<double>>(n);
    for(int i = 0; i<n; i++){
        input[i] = vector<double>(2);
        y[i] = vector<double>(3);
        double x1 = static_cast<double>(randInt(gen));
        double x2 = static_cast<double>(randInt(gen));
        input[i][0] = x1;
        input[i][1] = x2;

        //xor
        y[i][0] = ((static_cast<int>(input[i][0])+static_cast<int>(input[i][1]))%2);
        //or
        if(x1 || x2) y[i][1] = 1;
        //and
        if(x1 && x2) y[i][2] = x1;
        // cout << boolalpha << input[i][0] << input[i][1] << " " << y[i] << "\n";
    }




    vector<int> networkNodes = {2,2,3};
    int numLayers = 3;

    network nn;
    nn.create(numLayers, networkNodes);
    nn.display();


    // nn.calculate(inputs);
    nn.backprop(input, y, .1);
}