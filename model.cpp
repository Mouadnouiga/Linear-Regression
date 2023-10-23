#include "include/linreg.hpp"
#include <fstream>
#include <iostream>


int main( int argc, char *argv[] )
{
    if( argc < 3 || argc > 6 ){
        std::cerr << "Usage: linreg <traindata> <testdata> [alpha] [eps] [tmax]\n";
        exit(EXIT_FAILURE);
    }

    //  reading training data
    int m_train, n_train;
    float **x_train, *y_train;
    std::ifstream trainfile(argv[1]);
    if( trainfile.is_open() ){
        trainfile >> m_train >> n_train;
        x_train = new float*[m_train];
        y_train = new float[m_train];
        for (int i = 0; i < m_train; ++i){
            x_train[i] = new float[n_train];
            for (int j = 0; j < n_train; ++j)
                trainfile >> x_train[i][j];
                
            trainfile >> y_train[i];
        }
        trainfile.close();
    }

    //  reading testing data
    int m_test, n_test;
    float **x_test, *y_test;
    std::ifstream testfile(argv[2]);
    if( testfile.is_open() ){
        testfile >> m_test >> n_test;
        x_test = new float*[m_test];
        y_test = new float[m_test];
        for (int i = 0; i < m_test; ++i){
            x_test[i] = new float[n_test];
            for (int j = 0; j < n_test; ++j)
                testfile >> x_test[i][j];
                
            testfile >> y_test[i];
        }
        testfile.close();
    }

    float alpha = 0.00000001f; if( argc > 3) alpha = atof(argv[3]);
    float eps = 0.00000001f; if( argc > 4) eps = atof(argv[4]);
    uint32_t tmax = 1000000; if( argc > 5) tmax = atoi(argv[5]);

    linear::LinearRegression *model = new linear::LinearRegression();

    model->fit(x_train, y_train, m_train, n_train, tmax, alpha, eps, -0.2f, 0.2f);

    if(model->predict(x_test, m_test, "data/predictions.siam"))
        // std::cout << "predictions are in: data/predictions.siam\n";

    // float score = model->score(x_test, y_test, m_test, linear::MAE);
    // std::cout << "score: " << score << '\n',

    //  deallocate memory
    delete model;
    delete []y_test; delete []y_train;
    for( int i = 0; i < m_train; ++i )
        delete []x_train[i];
    delete []x_train;
    for( int i = 0; i < m_test; ++i )
        delete []x_test[i];
    delete []x_test;
}