import math


def read_from_txt(file_name):
    X = []
    y = []
    X_test = []
    y_test = []
    file = open(file_name, 'r')
    file_data = file.readlines()

    # grab info from head
    head_info = file_data[0].split(',')
    head_info[-1] = head_info[-1].replace('\n', '')
    dimension, radius = int(head_info[0]), int(head_info[-1])

    # input dataset
    for row in file_data[1:8001]:
        # split instance space, label space
        tmp_list = row.split(',')
        # replace return to blank
        tmp_list[-1] = tmp_list[-1].replace('\n', '')
        X.append(tmp_list[0:dimension])
        y.append(int(tmp_list[-1]))

    for row in file_data[8001:]:
        # split instance space, label space
        tmp_list = row.split(',')
        # replace return to blank
        tmp_list[-1] = tmp_list[-1].replace('\n', '')
        X_test.append(tmp_list[0:dimension])
        y_test.append(int(tmp_list[-1]))

    return X, y, X_test, y_test, dimension


def Gaussian_Kernel(vector1, vector2, dimension, sigma=3.0):
    L2_norm = 0
    for d in range(dimension):
        L2_norm += (float(vector1[d]) - float(vector2[d])) ** 2

    Kernel_Value = math.exp(- L2_norm / (2 * (sigma ** 2)))
    return Kernel_Value


def Fit(X_train, y_train, dimension):
    iteration = True
    vio_points = []
    vio_labels = []

    # The first sample point shall be violation  for sure
    vio_points.append(X_train[0])  # the first point is bound to be preserved
    vio_labels.append(y_train[0])
    iteration_num = 0
    while iteration:
        for sample_index, sample_point in enumerate(X_train[1:]):
            sample_label = y_train[sample_index + 1]
            total_kernel_value = 0
            for vio_index in range(len(vio_points)):
                kernel_value = Gaussian_Kernel(vector1=vio_points[vio_index],
                                               vector2=sample_point,
                                               dimension=dimension)
                # calculate accumulation
                total_kernel_value += kernel_value * vio_labels[vio_index]
            # Whether a new violation point
            if total_kernel_value * sample_label < 0:
                vio_points.append(sample_point)
                vio_labels.append(sample_label)
                break
            # Whether to stop iteration in case no violation
            if sample_index == len(X_train) - 2:
                iteration = False
        # Whether to stop iteration in case limit of time
        if iteration_num > 70:
            iteration = False
        iteration_num += 1

    return vio_points, vio_labels


def Predict(violation_points, violation_labels, test_data, dimension):
    total_m = 0
    for index in range(len(violation_points)):
        kernel_value = Gaussian_Kernel(vector1=violation_points[index], vector2=test_data, dimension=dimension)
        total_m += kernel_value * violation_labels[index]

    if total_m > 0:
        return 1
    else:
        return -1



def Accuracy(violation_points, violation_labels, X_test, y_test, dimension):
    correct = 0
    for sample_index, sample in enumerate(X_test):
        prediction = Predict(violation_points, violation_labels, test_data=sample, dimension=dimension)

        if prediction == y_test[sample_index]:
            correct += 1
    return correct / len(X_test)


def Gaussian_Kernel_Perceptron(file_list):
    # Get Dataset and parameters
    for file in file_list:
        print("Starting processing File:", file)
        X, y, X_test, y_test, dimension = read_from_txt(file)
        print("Model Fitting continuing……")
        Point_List, Label_List = Fit(X, y, dimension)
        print("Model Fitting finished!\n\n")
        print("Violation Point List is:", Point_List)
        print("Violation Point Label is:", Label_List)
        # print("Violation_Point_List:", Point_List)
        # print("Violation_Label_List:", Label_List)
        accuracy_score = Accuracy(Point_List, Label_List, X_test, y_test, dimension)
        print("Accuracy:", accuracy_score)
        print("\n\n")


File = ["d2r16n10000.txt",
        "d4r24n10000.txt",
        "d8r12n10000.txt"]
Gaussian_Kernel_Perceptron(File)
