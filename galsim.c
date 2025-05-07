#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "graphics/graphics.h"
#include <X11/Xlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>
#include <pthread.h>
#include <stdatomic.h>
#define eps 1e-3

//Reference: Lab5, Lab6, Lab7, Assignment2, chatgpt, deepseek
// Struct of Arrays (SOA)
typedef struct Particles {
    double *restrict x;      
    double * restrict y;      
    double * restrict mass;   
    double * restrict v_x;    
    double * restrict v_y;   
    double * restrict bright; 
} Particles;

typedef struct {
    Particles *particles; // Pointer to particle data (x, y, v_x, v_y, mass, bright)
    double **a_x_local, **a_y_local; // Shared local acceleration arrays       // Start and end indices for this thread
    int start, stride, N;                // Total number of particles
    double dt;            // Time step
    int thread_id;  // thrad Id 
} ThreadData;

// All threads wait until every thread has finished, learn from lab8
pthread_barrier_t barrier;


// Storing total acccerleration for all particles(global arrary), learn from openmp(reduction)
double *a_x, *a_y;

int NUM_THREADS = 12;

//measuring time, learn from Lab5_Task4
//Here we use inline to reduce functions calls,learn from Lab5_Task5
static inline double get_wall_seconds() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + (double)tv.tv_usec / 1000000;
}

// Read particles from a binary file, learn from Assignment2_part2
int readfile(const char *filename, Particles *particles, const int N) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        fprintf(stderr, "Cannot open input file: %s\n", filename);
        return 1;
    }
    for (int i = 0; i < N; i++) {
        if (fread(&particles->x[i], sizeof(double), 1, file) != 1 ||
            fread(&particles->y[i], sizeof(double), 1, file) != 1 ||
            fread(&particles->mass[i], sizeof(double), 1, file) != 1 ||
            fread(&particles->v_x[i], sizeof(double), 1, file) != 1 ||
            fread(&particles->v_y[i], sizeof(double), 1, file) != 1 ||
            fread(&particles->bright[i], sizeof(double), 1, file) != 1) {
            fprintf(stderr, "Error reading particle data at index %d\n", i);
            fclose(file);
            return 1;
        }
    }
    fclose(file);
    return 0;
}

// Simulation step, divide particles to different threads, learn from lab8 and chatgpt
// We implement a reduction operation in pthread, learn from openmp(reduction)
// 1. use local acceraltion array(a_x_local), a_x_local[thread_id] are local array for each thread,
//    Each thread computes acceleration  independently to avoid race conditions.
void* simulationStep(void *arg) {
    ThreadData *data = (ThreadData *)arg;
    Particles *particles = data->particles;
    int  start = data->start, stride = data->stride, N = data->N;
    double **a_x_local = data->a_x_local, **a_y_local = data->a_y_local;
    double dt = data->dt;
    int thread_id = data->thread_id;
    const double G = 100.0 / N;
  

    memset(a_x_local[thread_id], 0, N * sizeof(double));
    memset(a_y_local[thread_id], 0, N * sizeof(double));

    for (int i = start; i < N; i+=stride) {
        double tmpa_x = 0.0, tmpa_y = 0.0;
        double tmpx = particles->x[i];
        double tmpy = particles->y[i];
        const double tempmass = particles->mass[i];
        for (int j = i + 1; j < N; j++) {
            double r_x = tmpx - particles->x[j];
            double r_y = tmpy - particles->y[j];
            double r2 = r_x * r_x + r_y * r_y;
            double r = sqrt(r2);
            double r_eps = r + eps;
            double inv_r3 = 1.0 / (r_eps * r_eps * r_eps);
            double temp = -G * inv_r3;
            tmpa_x += temp * particles->mass[j] * r_x;
            tmpa_y += temp * particles->mass[j] * r_y;
            a_x_local[thread_id][j] -= temp * tempmass * r_x;
            a_y_local[thread_id][j] -= temp * tempmass * r_y;
        }
        a_x_local[thread_id][i] += tmpa_x;
        a_y_local[thread_id][i] += tmpa_y;
    }
    // 2. Using barrier ensures all thread finish computation
    pthread_barrier_wait(&barrier);
    // 3. local accerleration is summed into a_x, a_y(global arrary)
    for (int i = start; i < N; i+=stride) {
        for (int t = 0; t < NUM_THREADS; t++) {
            a_x[i] += a_x_local[t][i];
            a_y[i] += a_y_local[t][i];
        }
    }

    // updates velocity and position
    for (int i = start; i < N; i+=stride) {
        particles->v_x[i] += dt * a_x[i];
        particles->v_y[i] += dt * a_y[i];
        particles->x[i] += dt * particles->v_x[i];
        particles->y[i] += dt * particles->v_y[i];
    }
    pthread_exit(NULL);
}

// Write particles to a binary file
int binary_particle(const char *filename, const Particles *particles, const int N) {
    FILE *file = fopen(filename, "wb");
    if (file == NULL) {
        fprintf(stderr, "Cannot open output file: %s\n", filename);
        return 1;
    }
    for (int i = 0; i < N; i++) {
        if (fwrite(&particles->x[i], sizeof(double), 1, file) != 1 ||
            fwrite(&particles->y[i], sizeof(double), 1, file) != 1 ||
            fwrite(&particles->mass[i], sizeof(double), 1, file) != 1 ||
            fwrite(&particles->v_x[i], sizeof(double), 1, file) != 1 ||
            fwrite(&particles->v_y[i], sizeof(double), 1, file) != 1 ||
            fwrite(&particles->bright[i], sizeof(double), 1, file) != 1) {
                fprintf(stderr, "Error writing particle data at index %d\n", i);
                fclose(file);
                return 1;
            }
        }
        fclose(file);
        return 0;
}
    
int main(int argc, char *argv[]) {
    if (argc != 7) {
        printf("Usage: galsim N filename nsteps delta_t graphics n_threads\n");
        return 1;
    }
    
    const int N = atoi(argv[1]);
    const char *filename = argv[2];
    const int nsteps = atoi(argv[3]);
    const double dt = atof(argv[4]);
    const int graphicsEnabled = atoi(argv[5]);
    NUM_THREADS = atoi(argv[6]);
    int step = 0;
    
    // Allocate memory for the Particles struct (SOA)
    Particles particles;
    particles.x = (double*) malloc(N * sizeof(double));
    particles.y = (double*) malloc(N * sizeof(double));
    particles.v_x = (double*) malloc(N * sizeof(double));
    particles.v_y = (double*) malloc(N * sizeof(double));
    particles.mass = (double*) malloc(N * sizeof(double));
    particles.bright = (double*) malloc(N * sizeof(double));
    
    a_x = (double*)calloc(N, sizeof(double));
    a_y = (double*)calloc(N, sizeof(double));
    
    if (particles.x == NULL || particles.y == NULL || particles.v_x == NULL || 
        particles.v_y == NULL || particles.mass == NULL || particles.bright == NULL ||
        a_x == NULL || a_y == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
            return 1;
    }
        
    readfile(filename, &particles, N);
    double startTime1 = get_wall_seconds();
    pthread_t threads[NUM_THREADS];
    ThreadData thread_data[NUM_THREADS];
    pthread_barrier_init(&barrier, NULL, NUM_THREADS);
    
    // Allocate memory for local acceleration arrays
    double *a_x_local[NUM_THREADS], *a_y_local[NUM_THREADS];
    for (int t = 0; t < NUM_THREADS; t++) {
        a_x_local[t] = (double*) malloc(N * sizeof(double));
        a_y_local[t] = (double*) malloc(N * sizeof(double));
    }
        
    // Dynamic load balancing
    for (int t = 0; t < NUM_THREADS; t++) {
        thread_data[t].particles = &particles;
        thread_data[t].a_x_local = a_x_local;
        thread_data[t].a_y_local = a_y_local;
        thread_data[t].start = t;
        thread_data[t].stride=NUM_THREADS;
        thread_data[t].N = N;
        thread_data[t].dt = dt;
        thread_data[t].thread_id = t;
    }
    
    // Set up graphics
    const int windowWidth = 800;
    const int windowHeight = 600;
    
    // Main simulation loop
    for (step = 0; step < nsteps; step++) {
        // Clear global and local acceleration arrays
        memset(a_x, 0, N * sizeof(double));
        memset(a_y, 0, N * sizeof(double));
        
        
        for (int t = 0; t < NUM_THREADS; t++) {
            pthread_create(&threads[t], NULL, simulationStep, &thread_data[t]);
        }
        
        
        for (int t = 0; t < NUM_THREADS; t++) {
            pthread_join(threads[t], NULL);
        }
        
    }

    //when I use the input data's brightness, which is out of range, so we 
    // need to normalize brightness to [0, 1]
    if (graphicsEnabled == 1) {
        InitializeGraphics(argv[0], windowWidth, windowHeight);
        ClearScreen();
        float min_bright = 0.0f, max_bright = 1.0f;
        for (int i = 0; i < N; i++) {
            if (particles.bright[i] < min_bright) min_bright = particles.bright[i];
            if (particles.bright[i] > max_bright) max_bright = particles.bright[i];
        }
        for (int i = 0; i < N; i++) {
            particles.bright[i] = (particles.bright[i] - min_bright) / (max_bright - min_bright);
        }
        // Draw each particle as a circle.
        // Map particle coordinates (assumed in [0,1]) to screen coordinates, we learn it from chatgpt
        for (int i = 0; i < N; i++) {
            float screenX = (float)(particles.x[i] * windowWidth);
            float screenY = (float)(particles.y[i] * windowHeight);
            float radius = 5.0f * (float)particles.mass[i];
            float color = (float)particles.bright[i];
            DrawCircle(screenX, screenY, (float)windowWidth, (float)windowHeight, radius, color);
        }
        Refresh();
        //control the simulation speed, add a small delay, which comes from chatgpt
        usleep(20000);

    }

    if (graphicsEnabled == 1) {
        FlushDisplay();
        CloseDisplay();
    } else {
        printf("Simulation complete.\n");
    }

    // Write results to file
    binary_particle("result.gal", &particles, N);

    // Free memory
    free(particles.x);
    free(particles.y);
    free(particles.v_x);
    free(particles.v_y);
    free(particles.mass);
    free(particles.bright);
    free(a_x);
    free(a_y);
    for (int t = 0; t < NUM_THREADS; t++) {
        free(a_x_local[t]);
        free(a_y_local[t]);
    }

    pthread_barrier_destroy(&barrier);
    
    

    // Print total time taken
    double totalTimeTaken = get_wall_seconds() - startTime1;
    printf("totalTimeTaken = %f\n", totalTimeTaken);
    return 0;
}