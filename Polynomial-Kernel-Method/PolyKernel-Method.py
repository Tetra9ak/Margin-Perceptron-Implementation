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


#10.26 gyc:修改了函数名后，对下面每个函数的传入参数中加入多项式核的参数c，并在迭代函数中加入S1 S2的传入参数
#S1为标签为1的violation点的index集合，S2为标签为-1的violation点的index集合，

#基于之前的dot_product改写出了POLYkernel，可以计算k(p,q)=(pq+1)^c

#在Training_polykernel的训练开始时，将S1 S2初始化为空集，每迭代一个点对其内容进行更新；并且在更新w时进行修改
#迭代的具体修改见iteration_polykernel的函数中

#由于核方法的margin并没有讲过，我还不会，所以没有实现。直接按照最大迭代数停止即可，这个方法我得到的结果不是参数w，而是violation点集S1, S2. 并且在程序后随机找了训练集中的四个点进行测试。

def POLYkernel(vector1,vector2,polyKernel_c):
    len1 = len(vector1)
    value = 0
    for value_index in range(len1):
        value += float(vector1[value_index]) * float(vector2[value_index])

    value2=math.pow((value+1),polyKernel_c)

    return value2


def poly_WQ(point,S1,S2,polyKernel_c):
    SUMLIST1=[]
    SUMLIST2=[]
    for pts in S1:
        tmp=POLYkernel(point,pts,polyKernel_c)
        SUMLIST1.append(tmp)

    for pts in S2:
        tmp=POLYkernel(point,pts,polyKernel_c)
        SUMLIST2.append(tmp)
    wq=sum(SUMLIST1)-sum(SUMLIST2)
    return wq


def iteration_polykernel(X, y, w,S1,S2,dimension, gamma_guess,polyKernel_c):
    violation_exist = -1
    for index, point in enumerate(X):
        point_label = y[index]


        kernel_value = poly_WQ(point,S1,S2,polyKernel_c) #将点乘换为kernel
        if kernel_value != [0 for _ in range(dimension)]:
            if kernel_value < 0:
               kernel_label = -1
            else:
                kernel_label = 1

            # Whether a violation point or not
            # Condition1 : label of sample and dot product result have different sign
            # Condition2 : sample is so close to plane

            #更新S1，S2
            if ( (kernel_label * point_label < 0)):
                newpoint=list(map(float, X[index]))
                violation_exist = index

                if(point_label == 1):
                    S1.append(newpoint)
                    break
                else:
                    S2.append(newpoint)
                    break

            #margin_this=(dot_product_value*point_label)/(math.sqrt(sum(list(map(lambda x: x ** 2, w)))))
            #margin_this = (kernel_value * point_label) / (math.sqrt(sum(list(map(lambda x: x ** 2, w)))))
            # if (  margin_this< (gamma_guess / 2.0)
            #         or (kernel_label * point_label < 0)):

        else:
            violation_exist = index
            newpoint = list(map(float, X[index]))
            if (point_label == 1):
                S1.append(newpoint)
                break
            else:
                S2.append(newpoint)
                break


    return violation_exist,S1,S2,w


def Training_polykernel(X, y, w,S1,S2, max_epoch, dimension, radius,polyKernel_c):
    # initialize parameters
    gamma_guess = radius
    # loop for max_iteration times

    for index in range(max_epoch):
        #S1 stands for violation pts labeled 1
        #S2 stands for violation pts labeled -1
        violation,S1,S2, w = iteration_polykernel(X, y, w,S1,S2,dimension=dimension, gamma_guess=gamma_guess,polyKernel_c=polyKernel_c)
        if violation > -1:
            print("label of sample is", str(y[violation]))
            print("Violation point is", str(X[violation]))
            print("S1 length",len(S1))
            print("S2 length", len(S2))

            print('\n\n\n\n\n')
        else:
            return False
        return True



def Margin_Perceptron_polykernel(file_list,polyKernel_c):
    # Get Dataset and parameters
    for file in file_list:
        X, y, dimension, radius = read_from_txt(file)
        w = [0 for _ in range(dimension)]
        max_epochs = max_iteration_calculate(radius=radius, gamma_guess=radius)
        gamma_guess = radius
        S1 = []
        S2 = []

        while Training_polykernel(X, y, w,S1,S2,max_epoch=max_epochs, dimension=dimension, radius=gamma_guess,polyKernel_c=polyKernel_c):
            gamma_guess /= 2
            # print('gamma_guess:', gamma_guess)
            max_epochs = max_iteration_calculate(radius=radius, gamma_guess=gamma_guess)
            # print('max_epochs:', max_epochs)

            # if gamma_guess <= 1e-8:
            #     print('Due to approximate requirement, the Margin Perceptron stops')
            #     break

        print("poly kernel result:")
        print("set for label 1:",S1)
        print("set for label -1:", S2)


        sampled_indices = [1,10,1500,1145]
        sampled_X = [X[i] for i in sampled_indices]
        sampled_Y = [y[i] for i in sampled_indices]
        correctnumber=0
        for i in range(0,len(sampled_X)):
            testlabel= sampled_Y[i]
            kernel_value = poly_WQ(sampled_X[i], S1, S2, polyKernel_c)

            if(kernel_value*testlabel>0):
                correctnumber+=1

        print("4 point in train dataset's acc:",correctnumber)

        print('\n\n\n\n\n')




File = ["d2r16n10000.txt",
        "d4r24n10000.txt",
        "d8r12n10000.txt"]

polyKernel_c = 2

Margin_Perceptron_polykernel(File,polyKernel_c)





