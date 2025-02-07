import random as r
import math as m

weight = r.uniform(-1, 1)
bias= 0

training_data = [0.0, 1.0, 2.0, 3.0]
expected_values = [0.0, 5.0, 10.0, 15.0]



def yfunction(input):
    return weight *input + bias

def zfunction(input):
    return input

def MSE(output, expected_output):
    MSE=(output- expected_output)**2
    return MSE

def dMSE_DZ(output, expected_output):
    dmse_dz=2*(output-expected_output)
    return dmse_dz

def Neurona(input,expected_output,weight, bias):
    processed_x= yfunction(input)
    output= zfunction(processed_x)
    Mse= MSE(output, expected_output)

    print(f"output: {output}")
    print(f"loss: {Mse} ")

    DMSE_DZ = dMSE_DZ(output, expected_output)
    DZ_DY= 1.0
    DY_DW= input
    DY_DB = 1.0
    
    DMSE_dw =DMSE_DZ * DZ_DY * DY_DW
    DMSE_db = DMSE_DZ * DZ_DY * DY_DB

    
    Learning_Rate= 0.01
    weight -= DMSE_dw * Learning_Rate
    bias -= DMSE_db *Learning_Rate


def training(training_data, expected_values, epochs):
    global weight, bias  
    
    for epoch in range(epochs):
        total_loss = 0
        print(f"\nEpoch {epoch + 1}/{epochs}")
        

        for input_val, expected_val in zip(training_data, expected_values):

            processed_x = yfunction(input_val)
            output = zfunction(processed_x)
            loss = MSE(output, expected_val)
            total_loss += loss
            

            DMSE_DZ = dMSE_DZ(output, expected_val)
            DZ_DY = 1.0
            DY_DW = input_val
            DY_DB = 1.0
            
            DMSE_dw = DMSE_DZ * DZ_DY * DY_DW
            DMSE_db = DMSE_DZ * DZ_DY * DY_DB
            
            Learning_Rate = 0.01
            global weight, bias
            weight -= DMSE_dw * Learning_Rate
            bias -= DMSE_db * Learning_Rate

        avg_loss = total_loss / len(training_data)
        print(f"Average loss: {avg_loss}")
        print(f"Weight: {weight}, Bias: {bias}")

training(training_data, expected_values, epochs=1000)

def predict(input):
    processed_x= yfunction(input)
    output= zfunction(processed_x)
    output= m.ceil(output)
    print(f"Prediction Based On {input}: {output}")

predict(5.0)
