import numpy as np
# structure of the deep neural network - 1st layer: Input layer(1, 3), output layer(1, 1)

# activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return np.multiply(s, (1-s))
def f_forward(x, w1, b):
    # input to Output layer
    z = x.dot(w1)+b  # input of out layer
    a = sigmoid(z)  # output of out layer
    return a, z


# initializing the weights randomly
def generate_wt_bias(x, y):
    weights = []
    for i in range(x * y):
        weights.append(np.random.randn())
    weights_vector = (np.array(weights).reshape(x, y))
    b = np.random.randn()
    print(f"bias is {b}")
    return weights_vector, b

# for loss we will be using mean square error(MSE)
def loss(y, y_train):
    s = np.square(y-y_train)
    s = np.sum(s)/(2*len(y_train))
    return s


def train(x, y_train, w, b, alpha = 0.1, epoch = 500):
    acc= []
    for j in range(epoch):
        l = []
        for i in range(x.shape[0]):
            #back projection
            out, z = f_forward(x[i], w, b)
            l.append((loss(out, y_train[i])))
            dL_da = (out - y_train[i]) / len(y_train)
            da_dz = sigmoid_derivative(z)
            dz_dw = x[i]
            dL_dw = np.multiply(np.multiply(dL_da, da_dz), dz_dw)
            w -= alpha * np.reshape(dL_dw, (3,1))
            b -= alpha * dL_da * da_dz
        print("epochs:", j + 1, "======== acc:", (1 - (sum(l) / len(x))) * 100)
        acc.append((1 - (sum(l) / len(x))) * 100)
    return w, b
def predict(x, w, b):
    out, _ = f_forward(x, w, b)
    label = round(float(out))
    if label == 0:
        print("Image is labeled as 0.")
    elif label == 1:
        print("Image is labeled as 1.")
    return label


if __name__ == "__main__":

    x = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    labels = np.array([[0], [0], [1], [1]])

    w, bias = generate_wt_bias(3, 1)
    w_trained, bias_trained = train(x, labels, w, bias, 0.1, 500)

    for i in range(x.shape[0]):
        out = predict(x[i], w_trained, bias_trained)
        print(f"original label {labels[i]} while prediction is {out}")


