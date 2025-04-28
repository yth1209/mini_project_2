
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <pthread.h>
#include <time.h>
#include <chrono>
#include <functional>
#include <map>
#include <queue>
#include <atomic>

using namespace std;

void decomposed_A_to_L_U(std::vector<std::vector<double>>& A, vector<std::vector<double>>&L, vector<std::vector<double>>&U, int n);

void print_matrix(const std::vector<std::vector<double>>& mat, const char* name) {
    printf("%s:\n", name);
    for (const auto& row : mat) {
        for (double val : row)
            printf("%.4e ", val);
        printf("\n");
    }
}

// Sequential LU decomposition
void lu_decomposition_seq(std::vector<std::vector<double>>& A,
                          std::vector<std::vector<double>>& L,
                          std::vector<std::vector<double>>& U,
                          int n) {
    printf("Running sequential version...\n");
    auto t0 = std::chrono::steady_clock::now();

    vector<vector<double>> A_copy(A);

    for (int k = 0; k < n; k++) {
        // Pivoting
        double max = 0.0;
        int k_prime = k;
        
        // Find the maximum element in the k-th column
        for (int i = k; i < n; i++) {
            double abs_A_ik = abs(A_copy[i][k]);
            if (abs_A_ik > max) {
                max = abs_A_ik;
                k_prime = i;
            }
        }
    
        if (max == 0.0) {
            cerr << "Error: Singular matrix" << endl;
            exit(-1);
        }
    
        
        // Swap rows in P, A, and L
        if(k!=k_prime) {
            swap(A_copy[k], A_copy[k_prime]);
        }
    
        double pivot = A_copy[k][k];
        const vector<double>& A_k = A_copy[k];
    
        for (int i = k + 1; i < n; i++) {
            A_copy[i][k] = A_copy[i][k] / pivot;
            double L_ik = A_copy[i][k];
      
            auto& A_i = A_copy[i];
      
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


struct DecompArgs {
    double pivot;
    vector<vector<double>>& A_copy;
    int start;
    int end;
    int k;
    int n;
};

pthread_mutex_t decomp_lock = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t decomp_cond = PTHREAD_COND_INITIALIZER;
pthread_cond_t main_thread_cond = PTHREAD_COND_INITIALIZER;
queue<DecompArgs*> decomp_task_queue;
bool is_work_done = false;
atomic<int> remaining_tasks(0);

void* decomp_worker(void* arg){
    while(!is_work_done){
        pthread_mutex_lock(&decomp_lock);
        while(decomp_task_queue.empty()){
            pthread_cond_wait(&decomp_cond, &decomp_lock);
        }
        // cout << "Awake Pivot Worker Done " << endl;

        DecompArgs* args = decomp_task_queue.front();
        if(args == nullptr){
            pthread_mutex_unlock(&decomp_lock);
            continue;
        }
        decomp_task_queue.pop();
        pthread_mutex_unlock(&decomp_lock);
        
        for (int i = args->start; i < args->end; i++) {
            args->A_copy[i][args->k] /= args->pivot;
            double L_ik = args->A_copy[i][args->k];
      
            auto& A_i = args->A_copy[i];
      
            for (int j = args->k + 1; j < args->n; j++) {
                A_i[j] -= L_ik * args->A_copy[args->k][j];
            }
        }

        remaining_tasks--;
        delete args;

        pthread_cond_signal(&main_thread_cond);
    }

    return nullptr;
}

// Parallel LU decomposition using pthreads (stub)
void lu_decomposition_parallel(std::vector<std::vector<double>>& A,
                               std::vector<std::vector<double>>& L,
                               std::vector<std::vector<double>>& U,
                               int n, int t) {
    printf("Running Pthread version...\n");

    auto t0 = std::chrono::steady_clock::now();

    pthread_attr_t attr;
    pthread_attr_init(&attr);

    vector<vector<double>> A_copy(A);


    pthread_t pivot_threads[t];

    pthread_cond_init(&decomp_cond, NULL);
    pthread_cond_init(&main_thread_cond, NULL);
    pthread_mutex_init(&decomp_lock, NULL);

    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
    for(int i = 0; i < t; i++){
        pthread_create(&pivot_threads[i], &attr, decomp_worker, NULL);
    }

    for (int k = 0; k < n; k++) {
        // Pivoting
        double max_v = 0.0;
        int k_prime = k;

        // Find the maximum element in the k-th column
        for (int i = k; i < n; i++) {
            double abs_A_ik = abs(A_copy[i][k]);
            if (abs_A_ik > max_v) {
                max_v = abs_A_ik;
                k_prime = i;
            }
        }

        if (max_v == 0.0) {
            cerr << "Error: Singular matrix" << endl;
            cout << k << endl;
            exit(-1);
        }
    
        
        // Swap rows in P, A, and L
        if(k!=k_prime) {
            swap(A_copy[k], A_copy[k_prime]);
        }
    
        double pivot = A_copy[k][k];
        const vector<double>& A_k = A_copy[k];
    
        //define total number of tasks
        int task_num = t * 1;

        //calculate strid with ceiling
        int stride = max(1, (n-k-1 + task_num - 1)/task_num);
        for(int tid = 0; tid < task_num; tid++){
            int start = k+1 + (stride * tid);
            int end = min(n, start + stride);

            DecompArgs* args = new DecompArgs{pivot, A_copy, start, end, k, n};
            decomp_task_queue.push(args);

            remaining_tasks++;
        }
        pthread_cond_broadcast(&decomp_cond);

        pthread_mutex_lock(&decomp_lock);
        while(remaining_tasks>0){
            // pthread_cond_signal(&decomp_cond);
            pthread_cond_wait(&main_thread_cond, &decomp_lock);
        }
        pthread_mutex_unlock(&decomp_lock);

        if(!decomp_task_queue.empty()){
            cerr << "Error: Task queue not empty" << endl;
            exit(-1);
        }
    }

    is_work_done = true;

    pthread_cond_broadcast(&decomp_cond);

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

#ifdef PARALLEL
    lu_decomposition_parallel(A, L, U, n, t);
#else
    lu_decomposition_seq(A, L, U, n);
#endif

    if (p == 1) {
        print_matrix(L, "L");
        print_matrix(U, "U");
        print_matrix(A, "A");
    }

    return 0;
}


void decomposed_A_to_L_U(std::vector<std::vector<double>>& A, std::vector<std::vector<double>>&L, std::vector<std::vector<double>>&U, int n) {
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