// #include "ml.h"
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <random>
using namespace std;

float forward(float activation, float weight, float bias){
    float result = activation * weight + bias;
    return result;
}

// float backward(float activation, float weight, float bias, float learnRate, float y){
//     float gradient = (((activation * weight + bias)-y)*x);
//     return (w-learnRate*gradient);
// }

int main(){
    const int n = 100;
    vector<float> X(n);
    vector<float> y(n);
    
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> uniform_dist(-10.0,10.0);
    uniform_real_distribution<> randSlope(-10.0,10.0);
    uniform_real_distribution<> randIntercept(-20.0,20.0);
    normal_distribution<> noise_dist(0.0,1.0);

    float slope = randSlope(gen);
    float intercept = randIntercept(gen);

    for(int i = 0;i<n;i++){
        float x = uniform_dist(gen);
        float noise = noise_dist(gen);

        X[i] = x;
        y[i] = slope * x + intercept + noise;
        // cout << y[i] << "\n";
    }
    float w = 0;
    float b = 0;
    float loss = 0;
    float error;
    float y_hat = forward(X[0],w,b);
    float learningRate = .01;
    
    cout << "init error: " << y[0]-y_hat;
    cout << " init w: " << w << ", init b: " << b << "\n";
    for(int epoch = 0; epoch<500;epoch++){
        double dw = 0;
        double db = 0;
        double loss = 0;

        for(int k = 0;k<X.size();k++){
            y_hat = forward(X[k],w,b);
            error = y[k]-y_hat;
            loss += error * error;
            db += -2*error;
            dw += -2*error*X[k];
        }
        dw = dw/n;
        db = db/n;
        loss /= n;
        w -= learningRate*dw;
        b -= learningRate*db;
        cout << "Epoch " << epoch << " error: " << error << ", w: "<< w << ", b: " << b << "\n";
    }
    cout << "End error: " << error << ", End w: "<< w << ", End b: " << b << "\n";
    cout << "Slope: " << slope << ", Intercept: " << intercept;
}


