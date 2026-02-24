#include <vector>
#include <string>
#include <array>
#include <iostream>
#include <fstream>
#include <random>
#include <iomanip>
using namespace std;

double sigmoid(double x){
    return (1/(1+exp(-1*x)));
}

double dSigmoid(double x){
    return (sigmoid(x) * (1-sigmoid(x)));
}

int main(){
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> randInt(0,1);
    uniform_real_distribution<> randWeight(-.1,.1);
    array<array<double,2>,100> input;
    array<double,100> y;
    for(int i = 0; i<100; i++){
        double x1 = static_cast<double>(randInt(gen));
        double x2 = static_cast<double>(randInt(gen));
        input[i][0] = x1;
        input[i][1] = x2;
        y[i] = static_cast<double>((static_cast<int>(input[i][0])+static_cast<int>(input[i][1]))%2);
        // cout << boolalpha << input[i][0] << input[i][1] << " " << y[i] << "\n";
    }


    //initialize variables
    double lr = .2;

    array<array<double,2>,2> l1Weight;
    array<double,2> l1Bias = {0};
    array<double,2> z1Output = {0};

    array<double,2> l2Weight = {0};
    double l2Bias = 0;
    double z2Output = 0;

    double error = 0;
    double loss = 0;

    double y_hat = 0;
    double dy_hat = 0;
    double s2;
    array<double,2> dw2 = {0};
    array<double,2> da1 = {0};
    array<double,2> s1 = {0};
    array<double,2> a1 = {0};

    for(int i = 0; i<2;i++){
        l2Weight[i] = randWeight(gen);
        for(int k = 0;k<2;k++){
            l1Weight[i][k] = randWeight(gen);
        }
    }

    for(int epoch = 0; epoch < 1000; epoch++){
        double summation = 0;
        loss = 0;
        for(int i = 0;i<input.size(); i++){
            z1Output[0] = (l1Weight[0][0] * input[i][0]) + (l1Weight[0][1] * input[i][1]) + l1Bias[0];
            z1Output[1] = (l1Weight[1][0] * input[i][0]) + (l1Weight[1][1] * input[i][1]) + l1Bias[1];
            
            a1[0] = sigmoid(z1Output[0]);
            a1[1] = sigmoid(z1Output[1]);

            z2Output = (l2Weight[0] * a1[0]) + (l2Weight[1] * a1[1]) + l2Bias;
            y_hat = sigmoid(z2Output);
            summation += (y_hat-y[i]) * (y_hat-y[i]);

            dy_hat = (2)*(y_hat-(y[i]));
            s2 = dy_hat * dSigmoid(z2Output);
            da1[0] = s2 * l2Weight[0];
            da1[1] = s2 * l2Weight[1];
            dw2[0] = s2 * z1Output[0];
            dw2[1] = s2 * z1Output[1];
            
            s1[0] = (l2Weight[0] * s2) * (a1[0] * (1-a1[0]));
            s1[1] = (l2Weight[1] * s2) * (a1[1] * (1-a1[0]));

            l2Weight[0] -= lr * s2 * a1[0];
            l2Weight[1] -= lr * s2 * a1[1];
            l2Bias -= lr * s2;

            l1Weight[0][0] -= lr * s1[0] * input[i][0];
            l1Weight[0][1] -= lr * s1[0] * input[i][1];
            l1Weight[1][0] -= lr * s1[1] * input[i][0];
            l1Weight[1][1] -= lr * s1[1] * input[i][1];
            l1Bias[0] -= lr * s1[0];
            l1Bias[1] -= lr * s1[1];
            if(epoch == 999){
                cout << fixed << setprecision(6) << "x1: " << input[i][0] << ", x2: " << input[i][1] << ", outcome: " << y[i] << ", predicted: "  << y_hat << "\n";
            }
        }
        loss = summation/input.size();
        std::cout << "batch: " << epoch << " loss: " << loss << "\n";
    }
    std::cout << "end loss: " << loss;
    return 0;
}