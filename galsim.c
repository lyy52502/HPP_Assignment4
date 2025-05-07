#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "graphics/graphics.h"
#include <X11/Xlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>
#include <omp.h>

#define eps 1e-3

// Struct of Arrays (SOA)
typedef struct Particles {
    double *restrict x;      
    double * restrict y;      
    double * restrict mass;   
    double * restrict v_x;    
    double * restrict v_y;   
    double * restrict bright; 
} Particles;

// learn from lab10_task5
static inline double get_wall_seconds() {
    return omp_get_wtime();
}

//Read particles from a binary file, learn from Assignment2_part2
int readfile(const char *filename, Particles *particles, const int N) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
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
            fprintf(stderr, "Error reading particle data from file: %s\n", filename);
            fclose(file);
            return 1;
        }
    }
    fclose(file);
    return 0;
}



void simulationStep(Particles * restrict particles, double * restrict a_x, double * restrict a_y, const int N, const double dt) {
    const double G = 100.0 / N;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        a_x[i] = 0.0;
        a_y[i] = 0.0;
    }
    
    // Conbining with A3, and we use reduction and shcedule in the loop, here we used deepseek for improvement
    #pragma omp parallel for schedule(dynamic,10) reduction(+:a_x[:N], a_y[:N])
    for (int i = 0; i < N; i++) {
        const double tmpx = particles->x[i];
        const double tmpy = particles->y[i];
        const double tempmass = particles->mass[i];
        double tmpax = 0.0;
        double tmpay = 0.0;
        
        // We Vectorized inner loop
        #pragma omp simd reduction(+:tmpax,tmpay)
        for (int j = i+1; j < N; j++) {
            const double r_x = tmpx - particles->x[j];
            const double r_y = tmpy - particles->y[j];
            const double r_sq = r_x * r_x + r_y * r_y;
            const double r = sqrt(r_sq);
            const double r_e = (r + eps) * (r + eps) * (r + eps);
            const double temp = -G / r_e;
            const double fx = temp * r_x;
            const double fy = temp * r_y;
            
            tmpax += fx * particles->mass[j];
            tmpay += fy * particles->mass[j];
            a_x[j] -= fx * tempmass;
            a_y[j] -= fy * tempmass;
        }
        
        a_x[i] += tmpax;
        a_y[i] += tmpay;
    }
    
    
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        particles->v_x[i] += dt * a_x[i];
        particles->v_y[i] += dt * a_y[i];
    }

  
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        particles->x[i] += dt * particles->v_x[i];
        particles->y[i] += dt * particles->v_y[i];
    }
}



//Write particles to a binary file, which we use chatgpt to modify our code
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
    double startTime = get_wall_seconds();
    if (argc != 7) {
        printf("Usage: galsim N filename nsteps delta_t graphics n_threads\n");
        return 1;
    }
    
    const int N = atoi(argv[1]);
    const char *filename = argv[2];
    const int nsteps = atoi(argv[3]);
    const double dt = atof(argv[4]);
    const int graphicsEnabled = atoi(argv[5]);
    const int NUM_THREADS = atoi(argv[6]);
    omp_set_num_threads(NUM_THREADS);
    
    Particles particles;
    particles.x = (double*) malloc(N * sizeof(double));
    particles.y = (double*) malloc(N * sizeof(double));
    particles.v_x = (double*) malloc(N * sizeof(double));
    particles.v_y = (double*) malloc(N * sizeof(double));
    particles.mass = (double*) malloc(N * sizeof(double));
    particles.bright = (double*) malloc(N * sizeof(double));
    
    double *a_x = (double*) malloc(N * sizeof(double));
    double *a_y = (double*) malloc(N * sizeof(double));
    
    if (!particles.x || !particles.y || !particles.v_x || !particles.v_y || !particles.mass || !particles.bright || !a_x || !a_y) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }
    
    readfile(filename, &particles, N);
    
    const int windowWidth = 800, windowHeight = 600;
   
    
    for (int step = 0; step < nsteps; step++) {
        simulationStep(&particles, a_x, a_y, N, dt);
        
        if (graphicsEnabled) {
            InitializeGraphics(argv[0], windowWidth, windowHeight);
            ClearScreen();
            for (int i = 0; i < N; i++) {
                float screenX = (float)(particles.x[i] * windowWidth);
                float screenY = (float)(particles.y[i] * windowHeight);
                DrawCircle(screenX, screenY, (float)windowWidth, (float)windowHeight, 5.0f * (float)particles.mass[i], 1.0f);
            }
            Refresh();
            usleep(20000);
            if (CheckForQuit()) break;
        }
    }
    
    if (graphicsEnabled) {
        FlushDisplay();
        CloseDisplay();
    } else {
        printf("Simulation complete.\n");
    }
    // Write results to file
    binary_particle("result.gal", &particles, N);
    free(particles.x); 
    free(particles.y);
    free(particles.v_x);
    free(particles.v_y);
    free(particles.mass); 
    free(particles.bright);
    free(a_x); 
    free(a_y);
    
    printf("totalTimeTaken = %f\n", get_wall_seconds() - startTime);
    return 0;
}
