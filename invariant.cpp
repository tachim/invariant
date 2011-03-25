#include <Eigen/Eigen>
#include <cstdlib>

using namespace Eigen;

#define RANDF(a, b) ((a) + ((double) rand()) / RAND_MAX * ((b) - (a)))
#define MAX_INIT_WEIGHT .001

struct Derivatives {
    int n_out, n_hidden, n_output;

    Derivatives(int n_out_in, int n_hidden_in, int n_output_in)
        : n_out(n_out_in), n_hidden(n_hidden_in), n_output(n_output_in)
    {
    }
}

struct PlaneNNET {
    int imgdim, n_hidden, n_out;
    MatrixXd W1, W2;

    PlaneNNET(imgdim_in, n_hidden_in, n_out_in)
        : imgdim(imgdim_in), n_hidden(n_hidden_in), n_out(n_out_in)
    {
    }

    void init_random() {
        W1 = MatrixXf(imgdim + 1, n_hidden);
        W2 = MatrixXf(n_hidden + 1, n_out);

        for (int i = 0; i < imgdim + 1; i++)
            for (int j = 0; j < n_hidden; j++)
                W1(i, j) = RANDF(0, MAX_INIT_WEIGHT);

        for (int i = 0; i < n_hidden + 1; i++)
            for (int j = 0; j < n_out; j++)
                W2(i, j) = RANDF(0, MAX_INIT_WEIGHT);
    }
}

struct EvalNNet {
    MatrixXf inp, sh, xh, so, xo;
    EvalNNet(const MatrixXf &inp_in, const MatrixXf &sh_in, const MatrixXf &xh_in, 
            const MatrixXf &so_in, const MatrixXf &xo_in)
    {
        inp = inp_in;
        sh = sh_in;
        xh = xh_in;
        so = so_in;
        xo = xo_in;
    }
}
