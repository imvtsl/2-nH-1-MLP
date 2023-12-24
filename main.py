import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


def read_file(file_path):
    with open(file_path, 'r') as file:
        # Use numpy to load the data from the file
        coordinates_array = np.loadtxt(file)

    return coordinates_array


def normalize(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    normalized_data = (data - mean) / std
    return normalized_data


def load_and_normalise_data(file_name, label):
    x = read_file(file_name)

    x = normalize(x)

    # add labels to data
    y = None
    if label == 0:
        y = np.zeros((x.shape[0], 1), dtype='int')
    elif label == 1:
        y = np.ones((x.shape[0], 1), dtype='int')

    return np.hstack((x, y))


def load_training_and_validation_data():
    normalised_training_samples_class_1 = load_and_normalise_data(file_name="Train1.txt", label=0)
    normalised_training_samples_class_2 = load_and_normalise_data(file_name="Train2.txt", label=1)

    normalised_training_data_class_1 = normalised_training_samples_class_1[:1500, :]
    normalised_training_data_class_2 = normalised_training_samples_class_2[:1500, :]

    normalised_validation_data_class_1 = normalised_training_samples_class_1[1500:, :]
    normalised_validation_data_class_2 = normalised_training_samples_class_2[1500:, :]

    train = np.vstack((normalised_training_data_class_1, normalised_training_data_class_2))
    val = np.vstack((normalised_validation_data_class_1, normalised_validation_data_class_2))

    return train, val


def load_testing_data():
    test_1 = load_and_normalise_data(file_name="Test1.txt", label=0)
    test_2 = load_and_normalise_data(file_name="Test2.txt", label=1)

    test = np.vstack((test_1, test_2))
    return test


def initialize_weights_in_network(no_of_input, no_of_hidden, no_of_output):
    hidden_layer = [{'weights': np.random.rand(no_of_input + 1)} for _ in range(no_of_hidden)]
    output_layer = [{'weights': np.random.rand(no_of_hidden + 1)} for _ in range(no_of_output)]

    return [hidden_layer, output_layer]


def forward_propagation(neural_network, input_sample):
    augmented_input = np.hstack(([1], input_sample[:]))

    for layer in neural_network:
        layer_outputs = []
        for neuron in layer:
            neuron['output'] = max(0, np.dot(neuron['weights'], augmented_input))
            layer_outputs.append(neuron['output'])

        augmented_input = np.hstack(([1], layer_outputs[:]))
    return augmented_input[1:]


def transfer_derivative_function(value):
    return 1 if value > 0 else 0


def backward_propagation(neural_network, target_label):
    output_layer = neural_network[1]

    for neuron in output_layer:
        prediction_error = target_label - neuron['output']
        neuron['delta'] = prediction_error * transfer_derivative_function(neuron['output'])

    hidden_layer = neural_network[0]

    for n in range(len(hidden_layer)):
        error = 0.0
        for output_neuron in output_layer:
            error += output_neuron['weights'][n + 1] * output_neuron['delta']

        hidden_layer[n]['delta'] = error * transfer_derivative_function(hidden_layer[n]['output'])


def update_weights_in_network(neural_network, input_sample, learning_rate):
    augmented_input = np.hstack(([1], input_sample[:]))

    for layer in neural_network:
        layer_outputs = []
        for neuron in layer:
            neuron['weights'] += neuron['delta'] * augmented_input * learning_rate
            layer_outputs.append(neuron['output'])

        augmented_input = np.hstack(([1], layer_outputs[:]))


def calculate_error(neural_network, dataset, labels):
    total_error = 0.0
    for i, sample in enumerate(dataset):
        output = forward_propagation(neural_network, sample)
        ground_truth = labels[i]
        total_error += np.sum(np.square(output - ground_truth))

    return total_error / dataset.shape[0]


def file_name_format_string(n_input, n_hidden, n_output):
    return "MLP: {} x {} x {}".format(n_input, n_hidden, n_output)


def train_neural_net(neural_network, training_data, training_labels, validation_data, validation_labels, testing_data,
                     testing_labels,
                     learning_rate, max_iteration, no_of_input, no_of_hidden, no_of_output, parent_iteration):
    print("learning rate is:" + str(learning_rate))

    training_error_list = []
    validation_error_list = []
    testing_error_list = []
    iteration = 0
    validation_error = float('inf')
    is_training_finished = False

    while not is_training_finished:
        iteration += 1
        training_error = 0.0
        for i, sample in enumerate(training_data):
            ground_truth = training_labels[i]
            actual_output = forward_propagation(neural_network, sample)
            training_error += np.sum(np.square(actual_output - ground_truth))
            backward_propagation(neural_network, ground_truth)
            update_weights_in_network(neural_network, sample, learning_rate)

        training_error = training_error / training_data.shape[0]
        current_validation_error = calculate_error(neural_network, validation_data, validation_labels)
        testing_error = calculate_error(neural_network, testing_data, testing_labels)

        training_error_list.append(training_error)
        validation_error_list.append(current_validation_error)
        testing_error_list.append(testing_error)

        is_training_finished = np.isclose(validation_error, current_validation_error) or iteration >= max_iteration
        validation_error = current_validation_error
        print("iteration: {}; training_error: {}; validation_error: {}"
              .format(iteration, training_error, validation_error))

    print("Finished training")

    plt.title(file_name_format_string(no_of_input, no_of_hidden, no_of_output))
    plt.xlabel("Iteration Number")
    plt.ylabel("J/n")
    plt.plot(training_error_list, label="Training Error", color="red")
    plt.plot(validation_error_list, label="Validation Error", color="blue")
    plt.plot(testing_error_list, label="Testing Error", color="green")
    plt.legend()
    figure = plt.gcf()

    figure.savefig('./{}_{}_{}_{}.png'.format(parent_iteration, no_of_input, no_of_hidden, no_of_output))
    plt.close(figure)


def test_data_on_neural_net(network, data):
    outputs = []
    for sample in data:
        outputs.append(np.round(forward_propagation(network, sample)[0]))

    return np.array(outputs)


def train_and_evaluate_network(no_of_input, no_of_hidden, no_of_output, parent_iteration):
    neural_network = initialize_weights_in_network(no_of_input, no_of_hidden, no_of_output)

    # fetch and feed training data
    training_data, validation_data = load_training_and_validation_data()
    np.random.shuffle(training_data)
    x_train = training_data[:, :2]
    training_labels = training_data[:, 2]

    # fetch and feed validation data
    np.random.shuffle(validation_data)
    x_val = validation_data[:, :2]
    validation_labels = validation_data[:, 2]

    # fetch and feed testing data
    testing_data = load_testing_data()
    x_test = testing_data[:, :2]
    testing_labels = testing_data[:, 2]

    # train
    train_neural_net(neural_network, x_train, training_labels, x_val, validation_labels, x_test, testing_labels, 0.01, 1000,
                     no_of_input, no_of_hidden, no_of_output, parent_iteration)

    predicted_labels = test_data_on_neural_net(neural_network, x_test)
    accuracy = accuracy_score(testing_labels, predicted_labels)
    print("Accuracy on testing data: {}".format(accuracy))
    return accuracy


def project():
    average_accuracies = []
    for nh in range(2, 10 + 1, 2):
        print("begin training with {} hidden nodes".format(nh))
        accuracy_list = []
        for parent_iteration in range(10):
            accuracy = train_and_evaluate_network(2, nh, 1, parent_iteration)
            accuracy_list.append(accuracy)
        average_accuracy = np.mean(accuracy_list)
        average_accuracies.append(average_accuracy)
        print("finished training with {} hidden nodes".format(nh))
        print("average accuracy with {} hidden nodes is:{}".format(nh, average_accuracy))


if __name__ == '__main__':
    project()
