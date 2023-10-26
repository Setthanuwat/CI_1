import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from sklearn.metrics import confusion_matrix
import seaborn as sns

input_size = 2
hidden_size = 4
output_size = 2
learning_rate = 0.01
momentum_rate = 0.9
epochs = 10000
epsilon = 1e-4
num_folds = 10

def shuffled_data(normalized_data):
    shuffled_data = normalized_data.copy()
    np.random.shuffle(shuffled_data)
    output_file = "shuffled_data.txt"
    with open(output_file, "w") as file:
        for row in shuffled_data:
            line = " ".join(map(str, row)) 
            file.write(line + "\n")  

def read_data_from_text_file(file_path):
    data = []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            row_data = [float(num) for num in line.strip().split()]
            data.append(row_data)
   
    return data         
def normalize_data(data, max_val, min_val):
    normalized_data = []
    for row in data:
        normalized_row = [(value - min_val) / (max_val - min_val) for value in row]
        normalized_data.append(normalized_row)

    return normalized_data

def FindMaxMin(data):
    flat_data = [value for row in data for value in row]
    min_val = min(flat_data)
    max_val = max(flat_data)
    return max_val, min_val

def denormalize_data(data, max_val, min_val):
    denormalized_data = np.array(data) * (max_val - min_val) + min_val
    return denormalized_data


def setxy(normalized_data):
    X = [sublist[:input_size] for sublist in normalized_data]
    Y = [sublist[-output_size:] for sublist in normalized_data]
    return X, Y

def FindMaxMin_C(data):
    data_pairs = [pair for pair in data]
    max_pair = max(data_pairs)
    min_pair = min(data_pairs)

    max_valu = max(max_pair)
    min_valu = min(min_pair)
    max_values= max(max_valu)
    min_values= min(min_valu)
    return max_values,min_values

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def initialize_parameters(input_size, hidden_size, output_size):
    np.random.seed(1)
    hidden_layer_weights = np.random.uniform(size=(input_size, hidden_size))
    hidden_layer_bias = np.random.uniform(size=(1, hidden_size))
    output_layer_weights = np.random.uniform(size=(hidden_size, output_size))
    output_layer_bias = np.random.uniform(size=(1, output_size))
    return hidden_layer_weights, hidden_layer_bias, output_layer_weights, output_layer_bias


def backward_propagation(X, y, hidden_layer_output, predicted_output,hidden_layer_weights, hidden_layer_bias,output_layer_weights, output_layer_bias, learning_rate
                         , momentum_rate,prev_hidden_layer_weights, prev_hidden_layer_bias,prev_output_layer_weights, prev_output_layer_bias, epoch):
    #y = np.squeeze(y) 
   # y = y[:, np.newaxis]
    output_error = y - predicted_output

    mse = np.mean(output_error ** 2)
    # Calculate the Mean Squared Error
   # print(output_error)
   # if (epoch) % 10000 == 0:
       #print(f"Epoch: {epoch}, MSE: {mse}")
    #output_error =  output_error[0]
   # print(output_error)
    
    output_delta = output_error * sigmoid_derivative(predicted_output)
    hidden_layer_error = output_delta.dot(output_layer_weights.T)
    hidden_layer_delta = hidden_layer_error * sigmoid_derivative(hidden_layer_output)
    # Update the output layer weights and bias
    output_layer_weights_update = hidden_layer_output.T.dot(output_delta) * learning_rate + momentum_rate * (output_layer_weights - prev_output_layer_weights)
    output_layer_bias_update = np.sum(output_delta, axis=0, keepdims=True) * learning_rate + momentum_rate * (output_layer_bias - prev_output_layer_bias)

    # Update the hidden layer weights and bias
    hidden_layer_weights_update = X.T.dot(hidden_layer_delta) * learning_rate + momentum_rate * (hidden_layer_weights - prev_hidden_layer_weights)
    hidden_layer_bias_update = np.sum(hidden_layer_delta, axis=0, keepdims=True) * learning_rate + momentum_rate * (hidden_layer_bias - prev_hidden_layer_bias)

    # Update the weights and biases
    output_layer_weights += output_layer_weights_update
    output_layer_bias += output_layer_bias_update
    hidden_layer_weights += hidden_layer_weights_update
    hidden_layer_bias += hidden_layer_bias_update

    # Update the previous parameters for the next iteration
    np.copyto(prev_hidden_layer_weights, hidden_layer_weights)
    np.copyto(prev_hidden_layer_bias, hidden_layer_bias)
    np.copyto(prev_output_layer_weights, output_layer_weights)
    np.copyto(prev_output_layer_bias, output_layer_bias)
    denormalized_predicted_output = denormalize_data(predicted_output, Max, Min)
    denormalized_y = denormalize_data(y, Max, Min)
    return mse, denormalized_predicted_output, denormalized_y  



def forward_propagation(X, hidden_layer_weights, hidden_layer_bias, output_layer_weights, output_layer_bias):
    #print(X)
    hidden_layer_input = np.dot(X, hidden_layer_weights) + hidden_layer_bias
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, output_layer_weights) + output_layer_bias
    predicted_output = sigmoid(output_layer_input)
    #print(f"output :{output_layer_bias}")
    #print(predicted_output)
    
    return hidden_layer_output, predicted_output

def train_neural_network(X_train, Y_train, input_size, hidden_size, output_size, epochs, learning_rate, momentum_rate, epsilon):
    X_train = np.array(X_train)  
    Y_train = np.array(Y_train)  
    
    

    hidden_layer_weights, hidden_layer_bias, output_layer_weights, output_layer_bias = initialize_parameters(input_size, hidden_size,
                                                                                                             output_size)
    mse_values = []
    epochs_range = []
   
    prev_hidden_layer_weights = np.zeros_like(hidden_layer_weights)
    prev_hidden_layer_bias = np.zeros_like(hidden_layer_bias)
    prev_output_layer_weights = np.zeros_like(output_layer_weights)
    prev_output_layer_bias = np.zeros_like(output_layer_bias)

    for epoch in range(epochs):
        hidden_layer_output, predicted_output = forward_propagation(X_train, hidden_layer_weights, hidden_layer_bias
                                                                    , output_layer_weights, output_layer_bias)
        mse,denormalized_predicted_output,denormalized_y=backward_propagation(X_train, Y_train, 
                                                                              hidden_layer_output, predicted_output,
                             hidden_layer_weights, hidden_layer_bias,
                             output_layer_weights, output_layer_bias, learning_rate, momentum_rate,
                             prev_hidden_layer_weights, prev_hidden_layer_bias,
                             prev_output_layer_weights, prev_output_layer_bias, epoch)
        mse = np.mean((Y_train - predicted_output) ** 2)
        mse_values.append(mse) 
        epochs_range.append(epoch)  
        if mse < epsilon :
            print(f"Early stopping at Epoch {epoch} with MSE: {mse}")
            break
    return hidden_layer_weights, hidden_layer_bias, output_layer_weights, output_layer_bias

def calculate_percentage_error(y_true, y_pred):
    return 100-np.mean(np.abs((y_true - y_pred) / y_true)) * 100
def cross_validation(X, Y, num_folds, input_size, hidden_size, output_size, epochs, learning_rate, momentum_rate, epsilon,Max,Min):
    fold_size = len(X) // num_folds
    X = np.array(X)
    Y = np.array(Y)
    mse_scores = [] 
    for fold in range(num_folds):
        start_idx = fold * fold_size
        end_idx = (fold + 1) * fold_size
        
        X_test = X[start_idx:end_idx]
        Y_test = Y[start_idx:end_idx]
        
        X_train = np.concatenate((X[:start_idx], X[end_idx:]), axis=0)
        Y_train = np.concatenate((Y[:start_idx], Y[end_idx:]), axis=0)
        
        print(f"Fold {fold + 1}:")
        print(f"Test data size: {len(X_test)}")
        print(f"Train data size: {len(X_train)}")
        print(f"Hidden_size :{hidden_size}")
        print(f"Learning_rate :{learning_rate}")
        print(f"momentum_rate :{momentum_rate}")

        hidden_layer_weights, hidden_layer_bias, output_layer_weights, output_layer_bias = train_neural_network(X_train, Y_train, 
                                                 input_size, hidden_size, output_size, epochs, learning_rate, momentum_rate,epsilon)
        hidden_layer_output, predicted_output = forward_propagation(X_test, hidden_layer_weights, hidden_layer_bias, 
                                                                                            output_layer_weights, output_layer_bias)
        denormalized_predicted_output = denormalize_data(predicted_output, Max, Min)
        denormalized_Y_test = denormalize_data(np.array(Y_test)[:, np.newaxis], Max, Min)
        mse = np.mean((Y_test - predicted_output) ** 2)
        print(f"MSE for fold {fold + 1}: {mse}")
        
        mse_scores.append(mse)
        print("-" * 30)
    return mse_scores, denormalized_predicted_output,denormalized_Y_test

def ShowGraph(denormalized_predicted_output, denormalized_y):
    min_len = min(len(denormalized_predicted_output), len(denormalized_y))
    denormalized_predicted_output = denormalized_predicted_output[:min_len]
    denormalized_y = denormalized_y[:min_len]

    index = np.arange(min_len)

    bar_width = 0.3
    plt.figure(figsize=(10, 5))
    plt.bar(index, denormalized_predicted_output, width=bar_width, label='Predicted Output', color='blue', align='center')
    plt.bar(index + bar_width, denormalized_y, width=bar_width, label='Actual Output', color='red', alpha=0.5, align='edge')
    plt.xlabel('Data Point')
    plt.ylabel('Output Value')
    plt.title('Predicted Output vs. Actual Output')
    plt.xticks(index + bar_width / 2, [str(i) for i in range(min_len)])  # Label the bars with data point indices
    plt.legend()
    plt.grid()
    plt.show()
def confusion_matrix(predicted, actual):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    predicted = round_matrix(predicted, threshold=0.5)
    actual = round_matrix(actual, threshold=0.5)
    for i in range(len(predicted)):
            if actual[i][0] == 1 and predicted[i][0] == 1:
                TP += 1
            elif actual[i][0] == 0 and predicted[i][0] == 0:
                TN += 1
            elif actual[i][0] == 0 and predicted[i][0] == 1:
                FP += 1
            elif actual[i][0] == 1 and predicted[i][0] == 0:
                FN += 1
    print(f"True Positives (TP): {TP}")
    print(f"True Negatives (TN): {TN}")
    print(f"False Positives (FP): {FP}")
    print(f"False Negatives (FN): {FN}")
    accuracy = ((TP+TN)/(TP+TN+FP+FN))*100
    return TP, TN, FP, FN,accuracy
def round_matrix(matrix, threshold=0.5):
    rounded_matrix = np.where(matrix >= threshold, 1, 0)
    return rounded_matrix
def graphe_confusion_matrix(TP, TN, FP, FN):
    confusion_matrix = [[TN, FP], [FN, TP]]

    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = [0, 1]
    plt.xticks(tick_marks, ["Predicted [0 1]", "Predicted [1 0]"])
    plt.yticks(tick_marks, ["Actual [0 1]", "Actual [1 0]"])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(confusion_matrix[i][j]), horizontalalignment="center", color="white" if confusion_matrix[i][j] > (TP + TN + FP + FN) / 2 else "black")

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


data_table = read_data_from_text_file("cross.txt")
Max, Min = FindMaxMin(data_table)
normalized_data = normalize_data(data_table, Max, Min)
X_train, Y_train = setxy(normalized_data)
#print(X_train,Y_train)
#hidden_layer_weights, hidden_layer_bias, output_layer_weights, output_layer_bias = initialize_parameters(input_size, hidden_size, output_size)
mse_scores,denormalized_predicted_output,denormalized_y=cross_validation(X_train, Y_train, num_folds, input_size, hidden_size, output_size, epochs, learning_rate, momentum_rate, epsilon,Max,Min)
denormalized_y = np.squeeze(denormalized_y)
denormalized_predicted_output = np.squeeze(denormalized_predicted_output)
#print(denormalized_y)
#denormalized_y = denormalized_y[:, np.newaxis]
print(round_matrix(denormalized_predicted_output))
#print(denormalized_y)
#ShowGraph(denormalized_predicted_output,denormalized_y)
#accuracy = calculate_percentage_error(denormalized_y, denormalized_predicted_output)
#print(f"ความแม่นยำ: {accuracy :.2f}%")
#print(mse_scores)
#denormalized_y = np.squeeze(denormalized_y)
TP, TN, FP, FN,accuracy = confusion_matrix(denormalized_predicted_output, denormalized_y)
print(f"ความแม่นยำ: {accuracy:.2f}%")

graphe_confusion_matrix(TP, TN, FP, FN)
#print("ความแม่นยำ",compare_matrices(denormalized_predicted_output, denormalized_y, data_table))