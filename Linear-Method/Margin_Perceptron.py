import math


def read_from_txt(file_name):
    X = []
    y = []
    file = open(file_name, 'r')
    file_data = file.readlines()

    # grab info from head
    head_info = file_data[0].split(',')
    head_info[-1] = head_info[-1].replace('\n', '')
    dimension, radius = int(head_info[0]), int(head_info[-1])

    # input dataset
    for row in file_data[1:]:
        # split instance space, label space
        tmp_list = row.split(',')
        # replace return to blank
        tmp_list[-1] = tmp_list[-1].replace('\n', '')
        X.append(tmp_list[0:dimension])
        y.append(int(tmp_list[-1]))

    return X, y, dimension, radius


# calculate the maximum iteration
def max_iteration_calculate(radius, gamma_guess):
    max_iteration = math.ceil(12 * (radius ** 2) / (gamma_guess ** 2))
    return max_iteration


def dot_product(vector1, vector2):
    len1 = len(vector1)
    value = 0
    for value_index in range(len1):
        value += float(vector1[value_index]) * float(vector2[value_index])

    return value


def iteration(X, y, w, dimension, gamma_guess):
    violation_exist = -1
    for index, point in enumerate(X):
        point_label = y[index]

        dot_product_value = dot_product(w, point)
        if w != [0 for _ in range(dimension)]:
            if dot_product_value < 0:
                dot_product_label = -1
            else:
                dot_product_label = 1
            # Whether a violation point or not
            # Condition1 : label of sample and dot product result have different sign
            # Condition2 : sample is so close to plane
            if (abs(dot_product_value / (math.sqrt(sum(list(map(lambda x: x ** 2, w)))))) < (gamma_guess / 2.0)
                    or (dot_product_label * point_label < 0)):
                violation_exist = index
                break
        else:
            violation_exist = index
            break
    return violation_exist, w


def Training(X, y, w, max_epoch, dimension, radius):
    # initialize parameters
    gamma_guess = radius
    # loop for max_iteration times
    for index in range(max_epoch):
        violation, w = iteration(X, y, w, dimension=dimension, gamma_guess=gamma_guess)
        if violation > -1:
            print("w before updating", str(w))
            print("label of sample is", str(y[violation]))
            print("Violation point is", str(X[violation]))
            for i in range(dimension):
                # label = -1 -> w += -1 * p
                # label= 1 -> w += 1 * p
                w[i] += y[violation] * float(X[violation][i])
            print('w after updating', str(w))
            print('\n\n\n\n\n')
        else:
            return False
        return True


def Margin_Perceptron(file_list):
    # Get Dataset and parameters
    for file in file_list:
        X, y, dimension, radius = read_from_txt(file)
        w = [0 for _ in range(dimension)]
        max_epochs = max_iteration_calculate(radius=radius, gamma_guess=radius)
        gamma_guess = radius

        while Training(X, y, w, max_epoch=max_epochs, dimension=dimension, radius=gamma_guess):
            gamma_guess /= 2.0
            # print('gamma_guess:', gamma_guess)
            max_epochs = max_iteration_calculate(radius=radius, gamma_guess=gamma_guess)
            # print('max_epochs:', max_epochs)
            if gamma_guess <= 1e-8:
                print('Due to approximate requirement, the Margin Perceptron stops')
                break
        print('The final gamma_guess is:', gamma_guess)
        print('The final w found by margin perceptron is:', w)


File = ["d2r16n10000.txt",
        "d4r24n10000.txt",
        "d8r12n10000.txt"]
Margin_Perceptron(File)
