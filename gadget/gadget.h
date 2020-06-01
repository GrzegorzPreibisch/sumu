#include <cstdint>
#include <vector>

using namespace std;
using bitmap = uint64_t;
//using bitmap = unsigned long long int;

// these argument names must match those in swig interface file
// but not those in the cpp file

double weight_sum(double W, uint64_t *ivec, int d1, int d2, double *dvec, int nd, int n, vector<bitmap> U, vector<bitmap> T, int t_ub);

std::pair<double, std::vector<int>> weight_sum_contribs(double W, uint64_t *ivec, int d1, int d2, double *dvec, int nd, int n, vector<bitmap> U, vector<bitmap> T, int t_ub);

//double weight_sum(double W, uint64_t *ivec, int ni, double *dvec, int nd, int n, bitmap U, bitmap T, int t_ub);

//std::pair<double, std::vector<int>> weight_sum_contribs(double W, uint64_t *ivec, int ni, double *dvec, int nd, int n, bitmap U, bitmap T, int t_ub);


