#include "ml.h"
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
    float x = 2;
    float y = 4;
    float b = .5;
    float w = .5;
    float y_hat = forward(w,x,b);
    float l = pow(.5 * (y_hat-y),2);
    float error = y_hat-y;
    cout << "init error: " << error <<"\n";
    for(int i = 0; i < 100;i++){
        y_hat = forward(w,x,b);
        float gradient = ((y_hat-y)*x);
        w = (w-.1*gradient);
    }
    error = y_hat-y;
    cout << "End weight: " << w << "\nEnd error: "<< error;
}


