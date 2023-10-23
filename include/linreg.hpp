#pragma once


#include <math.h>


namespace linear
{
    typedef float _Tp;
    class ObjectFunction;
    class mse;
    class mae;
    class rmse;
    
    enum function{
        MSE = 0,
        MAE = 1,
        RMSE = 2
    };

    class LinearRegression{
        public:
            LinearRegression();

            ~LinearRegression();

            /**
             * @brief to train the model.
             * 
             * @param _x the feature matrix.
             * @param _y the output expected vector.
             * @param n_rows the numbre of samples in the dataset.
             * @param n_cols the numbre of features in data.
             * @param _t the number of iterations othrewise it equal 1000.
             * @param _eta the learning rate.
             * @param _eps the epsilon.
             * @param _min the minimum for initialization of weights.
             * @param _max the maximum for initialization of weights.
             */
            void fit(  _Tp **_x, _Tp *_y, int n_rows, int n_cols,
                             int _t=1000, _Tp _eta=0.00001, _Tp _eps=0.000001, _Tp _min=0., _Tp _max=0. );

            /**
             * @brief to get predictions
             * 
             * @param _x the feature matrix.
             * @param n_rows the number of sampels.
             * 
             * @return float*  
             */
            _Tp* predict( _Tp **_x, int n_rows ) const;

            /**
             * @brief to write predictions in a new .siam file
             *        
             * @param _x the feature matrix.
             * @param n_rows the number of sampels.
             * @param path path to the .siam file.
             * 
             * @return true if the predictions wrote correctely.
             * @return false if predictions does not wrotten correctely.
             */
            bool predict( _Tp **_x, int n_rows, const char* path ) const;

            /**
             * @brief to get predection.
             * 
             * @param _x the feature vector
             * @return float 
             */
            _Tp predict( _Tp *_x ) const;

            /**
             * @brief to calculate the score of the model.
             * 
             * @param _x testset features.
             * @param _y testset output expected.
             * @param error_function the object function to use. eg.linear::MSE, MAE, RMSE.
             * @return _Tp 
             */
            _Tp score( _Tp **_x, _Tp *_y, function error_function ) const ;

            /**
             * @brief to write the model parametres to a new .siam file. 
             * bias is the first element and the weights after. 
             * 
             * @param path path to the new .siam file.
             * @return true if the params wrote correctely.
             * @return false if params does not wrotten correctely. 
             */
            bool params_to_siam( const char *path );


            private:
            _Tp randf( _Tp _min, _Tp _max );
            void init_params( _Tp *, _Tp &, _Tp, _Tp );
            void learn( _Tp **, _Tp *, int, int );
            inline _Tp dot( _Tp *, _Tp *, int ) const ;
            void forward( _Tp **, int, int, _Tp * );
            void gradient_decent( _Tp **, _Tp *, _Tp *, int, _Tp * );
            void optimize(_Tp **, _Tp *, _Tp *, int);

            _Tp *m_W, m_B;
            int m;
            int n;
            _Tp eta;
            _Tp eps;
    };

    class ObjectFunction{
        public:
        virtual _Tp compute( _Tp *, _Tp *, int ) const = 0;
        virtual ~ObjectFunction(){}
    };

    class mse : public ObjectFunction{
        public:
        inline _Tp compute( _Tp *_y, _Tp *_yhat, int _n ) const override{
            _Tp error = (_y[0]-_yhat[0])*(_y[0]-_yhat[0]);

            for( int i = 1; i < _n; ++i ){
                linear::_Tp d = (_y[0]-_yhat[0]);
                error += d * d;
            }

            return error/(_Tp)_n;
        }
        ~mse(){}
    };

    class mae : public ObjectFunction{
        public:
        inline _Tp compute( _Tp *_y, _Tp *_yhat, int _n ) const override{
            linear::_Tp error = std::abs((_y[0]-_yhat[0]));

            for( int i = 1; i < _n; ++i )
                error += std::abs((_y[i]-_yhat[i]));

            return error/(_Tp)_n;
        }
        ~mae(){}
    };

    class rmse : public ObjectFunction{
        public:
        inline _Tp compute( _Tp *_y, _Tp *_yhat, int _n ) const override{
            linear::_Tp error = (_y[0]-_yhat[0])*(_y[0]-_yhat[0]);

            for( int i = 1; i < _n; ++i ){
                linear::_Tp d = (_y[0]-_yhat[0]);
                error += d * d;
            }

            return sqrtf(error/(_Tp)_n);
        }
        ~rmse(){}
    };
    
} // namespace linear
