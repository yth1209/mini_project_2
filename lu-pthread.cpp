
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

using namespace std;

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

    auto t1 = chrono::steady_clock::now();
    chrono::duration<double> elapsed = t1 - t0;
    cout << "Elapsed time: " 
         << elapsed.count() << " seconds\n";
}


struct PivotArgs {
    double& max_v;
    int& k_prime;
    vector<vector<double>>& A_copy;
    int start;
    int end;
    int k;
};

pthread_mutex_t pivot_lock = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t pivot_cond = PTHREAD_COND_INITIALIZER;
pthread_cond_t main_thread_cond = PTHREAD_COND_INITIALIZER;
queue<PivotArgs*> pivot_task_queue;
bool is_work_done = false;
int remaining_tasks = 0;

void* pivot_worker(void* arg){
    while(!is_work_done){
        pthread_mutex_lock(&pivot_lock);
        while(pivot_task_queue.empty()){
            pthread_cond_wait(&pivot_cond, &pivot_lock);
        }
        // cout << "Awake Pivot Worker Done " << endl;

        PivotArgs* args = pivot_task_queue.front();
        if(args == nullptr){
            pthread_mutex_unlock(&pivot_lock);
            continue;
        }
        pivot_task_queue.pop();
        pthread_mutex_unlock(&pivot_lock);
        
        double local_max = 0.0;
        int local_k_prime = args->k;

        for(int i = args->start; i < args->end; i++){
            double abs_A_ik = abs(args->A_copy[i][args->k]);
            if (abs_A_ik > local_max) {
                local_max = abs_A_ik;
                local_k_prime = i;
            }
        }

        pthread_mutex_lock(&pivot_lock);
        if(local_max !=0.0 && local_max > args->max_v){
            args->max_v = local_max;
            args->k_prime = local_k_prime;
        }
        remaining_tasks--;
        pthread_cond_signal(&main_thread_cond);
        pthread_mutex_unlock(&pivot_lock);
        delete args;
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

    pthread_cond_init(&pivot_cond, NULL);
    pthread_cond_init(&main_thread_cond, NULL);
    pthread_mutex_init(&pivot_lock, NULL);

    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
    for(int i = 0; i < t; i++){
        pthread_create(&pivot_threads[i], &attr, pivot_worker, NULL);
    }

    for (int k = 0; k < n; k++) {
        // Pivoting
        double max_v = 0.0;
        int k_prime = k;

        int stride = max(1, (n-k)/t);
        for(int tid = 0; tid < t; tid++){
            int start = k + (stride * tid);
            int end = min(n, start + stride);
    
            pthread_mutex_lock(&pivot_lock);
            PivotArgs* args = new PivotArgs{max_v, k_prime, A_copy, start, end, k};
            pivot_task_queue.push(args);

            remaining_tasks++;
            pthread_cond_signal(&pivot_cond);
            pthread_mutex_unlock(&pivot_lock);
        }


        pthread_mutex_lock(&pivot_lock);
        while(remaining_tasks>0){
            pthread_cond_signal(&pivot_cond);
            pthread_cond_wait(&main_thread_cond, &pivot_lock);
        }
        pthread_mutex_unlock(&pivot_lock);

        if(!pivot_task_queue.empty()){
            cerr << "Error: Task queue not empty" << endl;
            exit(-1);
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
    
        // for (int i = k + 1; i < n; i++) {
        //     A_copy[i][k] = A_copy[i][k] / pivot;
        //     double L_ik = A_copy[i][k];
      
        //     auto& A_i = A_copy[i];
      
        //     for (int j = k+1; j < n; j++) {
        //         A_i[j] -= L_ik * A_k[j];
        //     }
        // }
    }

    is_work_done = true;

    pthread_cond_broadcast(&pivot_cond);
    // for(int i = 0; i < t; i++){
    //     pthread_join(pivot_threads[i], NULL);
    // }

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
