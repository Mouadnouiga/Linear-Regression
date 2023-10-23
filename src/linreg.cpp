#include "../include/linreg.hpp"
#include <stdio.h>
#include <fstream>
#include <vector>


/********************************************************************************\
                                LinearRegression
\********************************************************************************/


linear::LinearRegression::LinearRegression()
{
    m_W = 0;
}

linear::LinearRegression::~LinearRegression()
{
    if(m_W)
        delete []m_W;
}


void 
linear::LinearRegression::fit( _Tp **_x, _Tp *_y, int n_rows, int n_cols,
                             int _t, _Tp _eta, _Tp _eps, _Tp _min, _Tp _max )
{
    eta = _eta; eps = _eps;
    m = n_rows;
    n = n_cols;
    m_W = new _Tp[n];
    init_params( m_W, m_B, _min, _max );

    learn( _x, _y, n, _t );
}


linear::_Tp *
linear::LinearRegression::predict( _Tp **_x, int _n ) const
{
    linear::_Tp *yhat = new _Tp[_n];
    for ( int i = 0; i < _n; ++i )
        yhat[i] = dot(_x[i], m_W, n) + m_B;

    return yhat;
}

bool linear::LinearRegression::predict(_Tp **_x, int n_rows, const char *path) const
{
    std::ofstream file(path, std::ios::out);
    if(file.is_open()){
        _Tp *yhat = predict(_x, n_rows);
        file << n_rows << '\n' << 1 << '\n';

        for( int i = 0; i < n_rows; ++i )
            file << yhat[i] << '\n';

        file.close();
        delete []yhat;
        return true;
    }
    return false;
}


linear::_Tp linear::LinearRegression::predict(_Tp *_x) const
{
    return dot(_x, m_W, n) + m_B;
}


linear::_Tp linear::LinearRegression::score( _Tp **_x, _Tp *_y, function _func = MSE ) const 
{
    linear::_Tp *yhat = this->predict(_x, m);
    ObjectFunction *Jw;
    switch (_func)
    {
        case MSE:
            Jw = new mse();
            break;
        case MAE:
            Jw = new mae();
            break;
        case RMSE:
            Jw = new rmse();
            break;

        default:
            perror("there is no object function.\n");
            exit(EXIT_FAILURE);
            break;
    }
    _Tp score = Jw->compute(_y, yhat, m);
    delete Jw;
    return score;
}


bool linear::LinearRegression::params_to_siam(const char *path)
{
    if(m_W){
        std::ofstream file(path ,std::ios::out);
        if(file.is_open()){
            file << n+1 << 1 << '\n';
            file << m_B << '\n';

            for( int i = 0; i < n; ++i )
                file << m_W[i] << '\n';

            file.close();
            return true;
        }
        return false;
    }
    return false;
}

/********************************************************************************\
                                Helping functions
\********************************************************************************/


linear::_Tp linear::LinearRegression::randf( _Tp _min, _Tp _max )
{
    return _min + rand() * (_max-_min)/RAND_MAX;
}


linear::_Tp linear::LinearRegression::dot( _Tp *_x, _Tp *_w, int _n ) const
{
    _Tp sum = _x[0] * _w[0];

    for( int i = 1; i < _n; ++i )
        sum += _x[i] * _w[i];

    return sum;
}


void linear::LinearRegression::init_params( _Tp *_w, _Tp& _b, _Tp _min, _Tp _max )
{
    for( int i = 0; i < n; ++i )
        _w[i] = randf(_min, _max);

    _b = randf( _min, _max );
}


void linear::LinearRegression::forward( _Tp **_x, int n_rows, int n_cols, _Tp *_dest )
{
    for( int i = 0; i < n_rows; ++i )
        _dest[i] = dot(_x[i], m_W, n_cols);
}


void linear::LinearRegression::gradient_decent( _Tp **_x, _Tp *_y, _Tp *_yhat, int n_cols, _Tp *dw )
{
    for( int j = 0; j < n_cols; ++j ){
        dw[j] = 0.0;
        for( int i = 0; i < m; ++i )
            dw[j] += (_y[i]-_yhat[i]) * _x[i][j];
    }
}


void linear::LinearRegression::optimize( _Tp **_x, _Tp *_y, _Tp *_yhat, int n_cols )
{
    _Tp *_dw = new _Tp[n_cols];
    gradient_decent(_x, _y, _yhat, n_cols, _dw);
    for ( int i = 0; i < n_cols; ++i )
        m_W[i] += eta * _dw[i];

    /* m_B: TODO */
    delete []_dw;
}


void linear::LinearRegression::learn( _Tp **_x, _Tp *_y, int n_cols, int _t)
{
    float *yhat = new float[m];
    forward(_x, m, n, yhat);
    
    _Tp *errors = new float[m];
    
    ObjectFunction *Jw = new mse();
    errors[0] = Jw->compute(_y, yhat, m);

    int t = 0;
    do{
        ++t;
        optimize(_x, _y, yhat, n_cols);
        forward( _x, m, n_cols, yhat ); 
        errors[t] = Jw->compute( _y, yhat, m );

    }while( errors[t-1]-errors[t] > eps && t < _t );

    delete []yhat;
    delete []errors;
    delete Jw;
}