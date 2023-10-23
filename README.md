# LINEAR REGRESSION using C++

### _this program is to perform a simple linear regression_

#### to use the program you can just type 
```bash
make
./linreg yourtraindata yourtestdata ...
```

#### or you can create a shared library to use it in your project if you want
```bash
make library
```

#### and to clean the object file you can type
```bash
make clean
```

#### then you can use `fit()` to train your linear regression model
#### and `predict()` to get preditions and you can save them in a file
#### also `score()` to get the score of your model using mse, mae or rmse objective functions.
#### or you can save your model parameters in a .siam file (it's a simple format we use it in our class)


## EXAMPLE

### This is a ploting for the error of the model in the model.cpp
![the error](file:///home/mouad/projects/education/siam/error_function.png)
