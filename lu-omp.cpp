
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <omp.h>
#include <chrono>
#include <functional>
#include <map>
#include <time.h>
#include <omp.h>

using namespace std;

vector<int> init_P(int n);
vector<std::vector<double>> copy_matrix(vector<vector<double>>& A, int n);

void decomposed_A_to_L_U(std::vector<std::vector<double>>& A, vector<std::vector<double>>&L, vector<std::vector<double>>&U, int n);


void print_matrix(const std::vector<std::vector<double>>& mat, const char* name) {
    printf("%s:\n", name);
    for (const auto& row : mat) {
        for (double val : row)
            printf("%.4e ", val);
        printf("\n");
    }
}

void lu_decomposition_omp(std::vector<std::vector<double>>& A,
                          std::vector<std::vector<double>>& L,
                          std::vector<std::vector<double>>& U,
                          int n, int t) {
    // TODO: implement LU decomposition using OpenMP
    printf("Running OpenMP version...\n");
    
    auto t0 = std::chrono::steady_clock::now();

    omp_set_num_threads(t);
    vector<int> P = init_P(n);
    vector<vector<double>> A_copy(A);

    for (int k = 0; k < n; k++) {
        // Pivoting
        double max = 0.0;
        int k_prime = k;
        
        // Find the maximum element in the k-th column
        #pragma omp parallel default(none) shared(max, k_prime, A_copy) firstprivate(k, n)
        { 
            // Initialize local variables for each thread
            double local_max = 0.0;
            int local_k_prime = k;
      
            #pragma omp for nowait schedule(dynamic)
            for (int i = k; i < n; i++) {
                double abs_A_ik = abs(A_copy[i][k]);
                if (abs_A_ik > local_max) {
                    local_max = abs_A_ik;
                    local_k_prime = i;
                }
            }
      
            // Update the global maximum and index if necessary
            #pragma omp critical
            {
                if (local_max > max) {
                    max = local_max;
                    k_prime = local_k_prime;
                }
            }
        }
    
        if (max == 0.0) {
            cerr << "Error: Singular matrix" << endl;
            exit(-1);
        }
    
        
        // Swap rows in P, A, and L
        if(k!=k_prime) {
            swap(P[k], P[k_prime]);
            swap(A_copy[k], A_copy[k_prime]);
        }
    
        double pivot = A_copy[k][k];
        const vector<double>& A_k = A_copy[k];
    
    
        #pragma omp parallel for default(none) shared(pivot, A_copy, A_k, k) firstprivate(n) schedule(dynamic)
        for (int i = k + 1; i < n; i++) {
            A_copy[i][k] = A_copy[i][k] / pivot;
            double L_ik = A_copy[i][k];
      
            auto& A_i = A_copy[i];
      
            #pragma omp simd
            for (int j = k+1; j < n; j++) {
                A_i[j] -= L_ik * A_k[j];
            }
        }
    }
    decomposed_A_to_L_U(A_copy, L, U, n);

    auto t1 = chrono::steady_clock::now();
    chrono::duration<double> elapsed = t1 - t0;
    cout << "Elapsed time: " 
         << elapsed.count() << " seconds\n";
}

int main(int argc, char* argv[]) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <n> <r> <t> <p>\n";
        return 1;
    }

    int n = atoi(argv[1]);
    int r = atoi(argv[2]);
    int t = atoi(argv[3]);
    int p = atoi(argv[4]);

    srand(r);
    std::vector<std::vector<double>> A(n, std::vector<double>(n));
    std::vector<std::vector<double>> L(n, std::vector<double>(n, 0.0));
    std::vector<std::vector<double>> U(n, std::vector<double>(n, 0.0));

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            A[i][j] = rand() / (double)RAND_MAX;

    lu_decomposition_omp(A, L, U, n, t);

    if (p == 1) {
        print_matrix(L, "L");
        print_matrix(U, "U");
        print_matrix(A, "A");
    }

    return 0;
}

  
vector<int> init_P(int n){
  vector<int> P(n);

  #pragma omp parallel default(none) shared(P) firstprivate(n)
  {
    #pragma omp for
    for (int i = 0; i < n; i++){
      P[i] = i;
    }
  }

  return P;
}
 
vector<std::vector<double>> copy_matrix(vector<std::vector<double>>& A, int n){
  std::vector<std::vector<double>> A_copy(n, std::vector<double>(n));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A_copy[i][j] = A[i][j];
        }
    }
    return A_copy;
}

void decomposed_A_to_L_U(std::vector<std::vector<double>>& A, std::vector<std::vector<double>>&L, std::vector<std::vector<double>>&U, int n) {
    // #pragma omp parallel for default(none) shared(A, L, U) firstprivate(matrix_size)
    for (int i = 0; i < n; i++) {
  
        for (int j = 0; j < n; j++) {
            if (i > j) {
                L[i][j] = A[i][j];
                U[i][j] = 0;
            } else if (i == j) {
                L[i][j] = 1;
                U[i][j] = A[i][j];
            } else {
                L[i][j] = 0;
                U[i][j] = A[i][j];
            }
        }
    }
}