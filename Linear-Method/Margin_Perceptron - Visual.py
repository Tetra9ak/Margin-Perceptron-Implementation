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
            margin_this=(dot_product_value*point_label)/(math.sqrt(sum(list(map(lambda x: x ** 2, w)))))
            # print('margin this:',margin_this)
            # print('gammaguess',gamma_guess / 2.0)
            if (  margin_this< (gamma_guess / 2.0)
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




#2D data perceptron visualization
import random
import matplotlib.pyplot as plt

def Visualization(file,X,y,w):
    if(file=="d2r16n10000.txt"):
        sampled_indices = random.sample(range(10000), 1000)

        sampled_X = [X[i] for i in sampled_indices]


        for items in sampled_X:
            items[0] = float(items[0])
            items[1] = float(items[1])

        sampled_Y = [y[i] for i in sampled_indices]

        colors = ['red' if y == 1 else 'blue' for y in sampled_Y]
        plt.scatter([x[0] for x in sampled_X], [x[1] for x in sampled_X], c=colors)
        # 绘制直线
        x = [-1, 1]
        y = [(w[0] * xi) / (-w[1]) for xi in x]
        max_len = max([abs(yi) for yi in y])
        y = [(yi / abs(yi)) * max_len for yi in y]
        plt.plot(x, y, color='yellow')

        # 显示图像
        plt.show()

        return 0
    else:
        return 0


def Margin_Perceptron(file_list):
    # Get Dataset and parameters
    for file in file_list:
        X, y, dimension, radius = read_from_txt(file)
        w = [0 for _ in range(dimension)]
        max_epochs = max_iteration_calculate(radius=radius, gamma_guess=radius)
        gamma_guess = radius

        while Training(X, y, w, max_epoch=max_epochs, dimension=dimension, radius=gamma_guess):
            gamma_guess /= 2
            # print('gamma_guess:', gamma_guess)
            max_epochs = max_iteration_calculate(radius=radius, gamma_guess=gamma_guess)
            # print('max_epochs:', max_epochs)
            Visualization(file,X,y,w)

            if gamma_guess <= 1e-8:
                print('Due to approximate requirement, the Margin Perceptron stops')
                break
        break
        print('The final gamma_guess is:', gamma_guess)
        print('The final w found by margin perceptron is:', w)






File = ["d2r16n10000.txt",
        "d4r24n10000.txt",
        "d8r12n10000.txt"]
Margin_Perceptron(File)





